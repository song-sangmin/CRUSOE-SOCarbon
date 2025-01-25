# os tools

import os.path

import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date, datetime, timedelta                 # for saving figures with today's date
import datetime
import gsw
import scipy
import mod_cremas as crx
import matplotlib
import matplotlib.pyplot     as plt  

import mod_ocean as myocn
import mod_plotting as myplot
import importlib
importlib.reload(myplot)
importlib.reload(myocn)

plt.rcParams.update(myplot.my_params(size=12))

# =============================================================================
# %% Common slicing functions
# =============================================================================

def get_profile_DS(floatDS, profid=None):
    """ 
    Function to return an xr Dataset with data from one vertical profile.
    @ param: floatDS (dataset with unique profid's as index)
    @ return: xr Dataset
    """
    if profid == None:
        profid = floatDS.profid[np.random.randint(0, len(floatDS.profid))].values.astype(str).tolist()
    return floatDS.sel({'profid':profid}), profid

def get_wmo_DS(floatDS, wmo=None):
    """ 
    Function to return an xr Dataset with data from one float.
    @ param: floatDS (dataset with wmoid's as coordinate, profids as index)
    @ return: wmo_DS (xr Dataset), wmo (float)
    """
    
    if wmo == None:
        all_wmoids = pd.DataFrame(floatDS.wmoid)[0].unique()
        wmo = all_wmoids[np.random.randint(0, len(all_wmoids))]

    wmo_DS = floatDS.sel(profid=[p for p in floatDS.profid.values if str(p)[:7] == str(wmo)[:7]])
    return wmo_DS, wmo

def get_n_random_profiles(floatDF, n=10):
    """
    @param  floatDF:  DataFrame with multiindex (profid, pressure)
            n:        Number of random profiles to select
    @return DataFrame with n random profiles 
    """
    profids = floatDF.reset_index().profid.unique()
    list_profids = np.random.choice(profids, n)
    
    return floatDF.loc[list_profids]


# =============================================================================
# %% Functions to process Argo data
# =============================================================================
def process_core_float(float_df, var_list = 'default', ref_time = '2014-01-01'):
    """
    Return dataframe from a single float dataset accessed with Argopy. 
    Assumed to be used in 'research' mode, i.e. already quality-controlled for core Argo. 

    @param:
            float_df (dict):  single float Dataframe, accessed in 'research' mode
            var_list : list of variables to include in the final dataframe, or 'default'
            ref_time : reference time for yearday calculation
    @return:
            float_df (pd.DataFrame): 
    """
    float_df = float_df.reset_index()
    float_df.rename(columns={'LATITUDE':'latitude','LONGITUDE':'longitude', 'TIME':'datetime', 'CYCLE_NUMBER':'cycle_number',
                            'PLATFORM_NUMBER':'wmoid', 'PRES':'pressure', 'TEMP':'temperature', 'PSAL':'salinity',
                            'PRES_ERROR': 'pres_error', 'PSAL_ERROR': 'psal_error', 'TEMP_ERROR': 'temp_error',}, inplace=True)
    float_df['yearday'] = myocn.datetime2ytd(float_df['datetime'], ref_time = ref_time)

    # Create a 10-digit unique id so easy to sort later
    # Make sure strings are filled so 1st and 10th profile are different
    prof = [tag.zfill(3) for tag in float_df['cycle_number'].astype(str)]
    float_df['profid'] = [str(float_df.wmoid[0]) + '_id' + tag for tag in prof]

    # Add calculated variables using gsw
    float_df['SA']= gsw.SA_from_SP(float_df['salinity'],float_df['pressure'],float_df['longitude'],float_df['latitude'])
    float_df['CT'] = gsw.CT_from_t(float_df['SA'], float_df['temperature'], float_df['pressure']) 
    float_df['sigma0'] = gsw.sigma0(float_df.SA.values, float_df.CT.values)
    float_df['spice'] = gsw.spiciness0(float_df["SA"].values, float_df["CT"].values)

    if var_list == 'default':
            var_list = ['wmoid', 'profid', 'latitude', 'longitude', 'datetime', 'yearday',
            'pressure', 'CT', 'SA', 'sigma0', 'spice',
            'temperature', 'salinity',
            'temp_error', 'psal_error', 'pres_error', 'cycle_number']

    float_df = float_df[var_list]

    return float_df

def interpolate_float_pressure(argodat, pres_levels = [0,1002,2], var_list = 'phys', ref_time = '2014-01-01'):
    ''' 
    Interpolate variables to a regular pressure grid.
    @ param  argodat: dataframe with "profid" as a variable column
                      can hold profiles from one float, or from multiple floats
             pres_levels: regular pressure levels to interpolate to
             var_list: list of variables to interpolate, or 'phys' for core Argo variables
    @ return argodat_regular: dataframe with interpolated variables
                            can be converted to xr Dataset with to_xr_dataset()
    '''
    if var_list == 'phys':
        var_list = ['CT', 'SA', 'sigma0', 'spice', 'temperature', 
                    'salinity', 'yearday', 'latitude', 'longitude']

    # # Initialize a list of dataframes, one for each profid, with interpolated values
    profile_interp_list = []
    interp_dict_perProfile = {k:None for k in var_list}

    for profid, group in argodat.groupby('profid'):
        # print('Processing profile:', profid)
        group = group.dropna(subset=['CT']).sort_values(by='pressure').reset_index()
        group = group.drop_duplicates(subset='pressure', keep="last") # why are there pressure duplicates? 

        for var in var_list:
            try: 
                f = scipy.interpolate.PchipInterpolator(x=group['pressure'], y=group[var], extrapolate = False)
                interp_dict_perProfile[var] = f(pres_levels)
            except: 
                interp_dict_perProfile[var] = np.tile(np.nan, len(pres_levels))
        
        # Single profile, interpolated
        prof_interp = pd.DataFrame({var: interp_dict_perProfile[var] for var in var_list})
        prof_interp['profid'] = profid
        prof_interp['wmoid'] = group['wmoid'].values[0]
        prof_interp['pressure'] = pres_levels

        profile_interp_list.append(prof_interp)

    argodat_regular = pd.concat(profile_interp_list).dropna(subset=['CT', 'SA'])
    argodat_regular = argodat_regular.set_index(["profid", 'pressure'])
    argodat_regular['datetime'] = pd.to_datetime(myocn.ytd2datetime(argodat_regular['yearday'], ref_time = ref_time))

    argodat_regular = argodat_regular.dropna(subset=['CT', 'SA'])

    return argodat_regular

def to_xr_dataset(argo_DF, title, source):
    """ 
    Convert float Dataframe to Dataset
    """
    argo_DS = xr.Dataset.from_dataframe(argo_DF)
    argo_DS = argo_DS.set_coords([ 'wmoid', 'datetime', 'yearday', 'latitude', 'longitude'])
    argo_DS = argo_DS.assign_attrs({'title':title, 'source': source, 'date':str(datetime.datetime.now())})
    return argo_DS
    

def print_float_bounds(DF):
    # print('Bounds of data: \n')
    print('Dates: \t\t' + str(DF.datetime.min()) + ' to ' + str(DF.datetime.max()))
    print('Latitude:\t' + str(DF.latitude.min()) + ' to ' + str(DF.latitude.max()))
    print('Longitude:\t' + str(DF.longitude.min()) + ' to ' + str(DF.longitude.max()))
# BGC_DF['profid'] = BGC_DF['profid'].apply(lambda x: f"{x[:7]}_id{x[7:]}")


# =============================================================================
# %% Plotting
# =============================================================================

from   cmocean               import cm as cmo
import cartopy
import cartopy.crs           as     ccrs   

# Prepare for plotting histogram of profile locations
# Longitude and latitude range
lon_min = -180
lon_max =  180
lat_min = -85
lat_max = -30

# depth range
zmin = 20.0
zmax = 1000.0

# ranges
lon_range   = (lon_min, lon_max)
lat_range   = (lat_min, lat_max)
depth_range = (zmin, zmax)



def plot_profiles_from_WMO(floatDF, var = 'CT', ax=None, dotsize=2, ylims=[1600,0]):
    """  
    See a selection of profiles as fxn of depth. 
    @param: floatDF: dataframe, either with 'profid' column (multiple profiles) 
                    or representing a single profile
    """
    if ax == None:
        fig = plt.figure(figsize=(6, 8))
        ax = fig.gca()

    floatDF = floatDF.reset_index()
    if 'profid' in floatDF.columns: # If floatDF has multiple profiles
        for name, group in floatDF.groupby('profid'):
            ax.scatter(group[var], group['pressure'], label=name, s=dotsize, marker='.', alpha=0.3, zorder=3)
    else: # If you indexed by profid (e.g. floatDF.loc['5906030_id010'])
        ax.scatter(floatDF[var], floatDF['pressure'], s=dotsize, marker='.', alpha=0.3, zorder=3)

    ax.invert_yaxis()
    ax.set_ylim(ylims)
    ax.set_ylabel('pressure')
    ax.set_xlabel(var)
    ax.grid(alpha=0.2, zorder=1)

    return ax


def plot_TS_diagnostics(wmo_DF, figsize=(12,8), dot_small=2, dot_large = 14):
    """ 
    Plot diagnostic figures for a single WMO dataframe.
    @param: wmo_DF: dataframe with 'profid' column
    @return: fig, axs
            ax1,2 : Sections of CT, SA
            ax3  : Map of float locations
            ax4,5 : Profiles of CT, SA
            ax6   : T-S diagram
    """
    wmo_DF = wmo_DF.reset_index()
    my_wmo = wmo_DF['wmoid'][0]
    print_float_bounds(wmo_DF)
    # ==== 

    fig  = plt.figure(figsize=figsize, layout='tight')
    # axs = axs.flatten()
    gs = fig.add_gridspec(3,3)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax3 = fig.add_subplot(gs[2, 0:2], projection = ccrs.PlateCarree())

    ax4 = fig.add_subplot(gs[0, 2])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 2])

    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313, projection = ccrs.PlateCarree())

    for ax in [ax1]:
        sca_CT = ax.scatter(wmo_DF.datetime, wmo_DF.pressure, c=wmo_DF.CT, cmap=cmo.thermal)
        plt.colorbar(sca_CT)
        ax.invert_yaxis()
        ax.set_ylabel('pressure')
        ax.set_title('Float ' + str(my_wmo)[:7] + ', CT')

    for ax in [ax2]:
        sca_SA = ax.scatter(wmo_DF.datetime, wmo_DF.pressure, c=wmo_DF.SA, cmap=cmo.haline)
        plt.colorbar(sca_SA)
        ax.invert_yaxis()
        ax.set_ylabel('pressure')
        ax.set_title('Float ' + str(my_wmo)[:7] + ', SA')

    for ax in [ax3]:
        sca_map = ax.scatter(wmo_DF.longitude, wmo_DF.latitude, c='r', s=dot_large, transform=ccrs.PlateCarree())
        ax.coastlines(resolution='50m',color='gray')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='gray', alpha=0.5, linestyle='--')
        ax.add_feature(cartopy.feature.LAND)

        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], ccrs.PlateCarree())

        # plt.colorbar(sca_map)

    for ax in [ax4]:
        plot_profiles_from_WMO(wmo_DF, var='CT', ax=ax, dotsize=dot_small)
        ax.set_ylim([1000, 0])
        ax.set_title('Temperature Profiles')

    for ax in [ax5]:
        plot_profiles_from_WMO(wmo_DF, var='SA', ax=ax, dotsize=dot_small)
        ax.set_ylim([1000, 0])
        ax.set_title('Salinity Profiles')

    for ax in [ax6]:
        myplot.setup_TS_contours(wmo_DF, ax6, contour_font_size=10)
        ax.scatter(wmo_DF.SA, wmo_DF.CT, c='r', s=dot_small, alpha=0.5)
    
    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]