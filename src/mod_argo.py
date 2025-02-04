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


# %% Common slicing functions
# =============================================================================

def get_profile_DS(argoDS, profid=None):
    """ 
    Function to return an xr Dataset with data from one vertical profile.
    @ param: argoDS (dataset with unique profid's as index)
    @ return: xr Dataset
    """
    if profid == None:
        profid = argoDS.profid[np.random.randint(0, len(argoDS.profid))].values.astype(str).tolist()
    return argoDS.sel({'profid':profid}), profid

def get_wmo_DS(argoDS, wmo=None):
    """ 
    Function to return an xr Dataset with data from one argo.
    @ param: argoDS (dataset with wmoid's as coordinate, profids as index)
    @ return: wmo_DS (xr Dataset), wmo (argo)
    """
    if wmo == None:
        all_wmoids = pd.DataFrame(argoDS.wmoid)[0].unique()
        wmo = all_wmoids[np.random.randint(0, len(all_wmoids))]
    wmo_DS = argoDS.sel(profid=[p for p in argoDS.profid.values if str(p)[:7] == str(wmo)[:7]])
    return wmo_DS, wmo

def get_n_random_profiles(argoDF, n=10):
    """
    @param  argoDF:  DataFrame with multiindex (profid, pressure)
            n:        Number of random profiles to select
    @return DataFrame with n random profiles 
    """
    profids = argoDF.reset_index().profid.unique()
    return argoDF.loc[np.random.choice(profids, n)]


def print_float_bounds(argo_DF):
    # print('Bounds of data: \n')
    print('Dates: \t\t' + str(argo_DF.datetime.min()) + ' to ' + str(argo_DF.datetime.max()))
    print('Latitude:\t' + str(argo_DF.latitude.min()) + ' to ' + str(argo_DF.latitude.max()))
    print('Longitude:\t' + str(argo_DF.longitude.min()) + ' to ' + str(argo_DF.longitude.max()))

# BGC_DF['profid'] = BGC_DF['profid'].apply(lambda x: f"{x[:7]}_id{x[7:]}")


# %% Functions to process Argo data (use with Argopy)
# =============================================================================

# Modify to work for bgc-floats, jan 29
# this is a modified version of mod_argo function, process_core_float()

def process_argo_float(float_df, bgc_list = ['pH'], ref_time = '2014-01-01'):
    """
    Return dataframe from a single core or BGC float dataset, accessed with Argopy. 
    Assumed to be used in 'expert' mode, i.e. not quality-controlled yet for BGC.
    (From Argopy, download and use .point2profile() and then .to_dataframe() to get input float dataframe)

    @param:
            float_df (dict):  single float Dataframe, accessed in 'research' mode for core
                                only 'expert' mode available for BGC floats
            bgc_list : list of BGC variables to include in the final dataframe
                        default ['pH'] adds only pH data
            ref_time : reference time for yearday calculation
    @return:
            float_df (pd.DataFrame): 
    """
    float_df = float_df.reset_index()


    # Default columns to rename, starting with necessary properties across core/bgc
    # Note that Argopy "research mode" has removed "ADJUSTED" from column names
    new_columns = {'LATITUDE':'latitude','LONGITUDE':'longitude', 'TIME':'datetime', 
                'CYCLE_NUMBER':'cycle_number', 'PLATFORM_NUMBER':'wmoid', 
                'PRES_ADJUSTED':'pressure', 'TEMP_ADJUSTED':'temperature', 'PSAL_ADJUSTED':'salinity'}
    # Rename QC and error columns
    new_columns.update({'TIME_QC': 'time_qc', 'POSITION_QC': 'position_qc', 
                        'PRES_ADJUSTED_QC': 'pressure_qc', 
                        'TEMP_ADJUSTED_QC': 'temperature_qc','PSAL_ADJUSTED_QC': 'salinity_qc'})
    new_columns.update({'PRES_ADJUSTED_ERROR': 'pres_error', 
                        'PSAL_ADJUSTED_ERROR': 'psal_error', 'TEMP_ADJUSTED_ERROR': 'temp_error'})
    
    # output_vars = new_columns.values()

    # ==================
    # Add BGC variables to the new column names
    if 'pH' in bgc_list: # expert mode
        new_columns.update({'PH_IN_SITU_TOTAL_ADJUSTED': 'pH', 'PH_IN_SITU_TOTAL_ADJUSTED_QC': 'pH_qc',
                            'PH_IN_SITU_TOTAL_ADJUSTED_ERROR': 'pH_error'})
    if 'oxygen' in bgc_list: 
        new_columns.update({'DOXY_ADJUSTED': 'oxygen', 'DOXY_ADJUSTED_QC': 'oxygen_qc',
                            'DOXY_ADJUSTED_ERROR': 'oxygen_error'})
    # if 'nitrate' in bgc_list:
    #     new_columns.update({'NITRATE_ADJUSTED': 'nitrate', 'NITRATE_ADJUSTED_QC': 'nitrate_qc',
    #                         'NITRATE_ADJUSTED_ERROR': 'nitrate_error'})
    # ==================
        
    float_df.rename(columns=new_columns, inplace=True)
    float_df['yearday'] = myocn.datetime2ytd(float_df['datetime'], ref_time = ref_time)

    # Create a unique profile id to be a useful index
    # Make sure strings are filled so 1st and 10th profile are different
    prof = [tag.zfill(3) for tag in float_df['cycle_number'].astype(str)]
    float_df['profid'] = [str(float_df.wmoid[0]) + '_id' + tag for tag in prof]

    # Add calculated variables using gsw
    float_df['SA']= gsw.SA_from_SP(float_df['salinity'],float_df['pressure'],float_df['longitude'],float_df['latitude'])
    float_df['CT'] = gsw.CT_from_t(float_df['SA'], float_df['temperature'], float_df['pressure']) 
    float_df['sigma0'] = gsw.sigma0(float_df.SA.values, float_df.CT.values)
    float_df['spice'] = gsw.spiciness0(float_df["SA"].values, float_df["CT"].values)

    # Turn all QC flags into strings
    qc_vars = [var for var in float_df.columns.tolist() if '_qc' in var]
    for k in qc_vars:
        float_df[k] = float_df[k].astype(str)


    # Standard variable list to return (core)
    # Can reorder by changing the output_vars list 
    output_vars = ['wmoid', 'profid', 'latitude', 'longitude', 'datetime', 'yearday',
            'pressure', 'CT', 'SA', 'sigma0', 'spice',
            'temperature', 'salinity',
            'temperature_qc', 'salinity_qc', 'pressure_qc',
            'time_qc', 'position_qc',
            'temp_error', 'psal_error', 'pres_error']
    
    for x in bgc_list:
        output_vars = output_vars + [x, x+'_qc', x+'_error']

    return float_df[output_vars]


# def process_core_float(float_df, var_list = 'phys', ref_time = '2014-01-01'):
#     """
#     (REPLACING WITH GENERAL FUNCTION, process_argo_float(). Outdated Jan 30 2025)
#     Return dataframe from a single float dataset accessed with Argopy. 
#     Assumed to be used in 'research' mode, i.e. already quality-controlled for core Argo. 

#     @param:
#             float_df (dict):  single float Dataframe, accessed in 'research' mode
#             var_list : list of variables to include in the final dataframe, or 'default'
#             ref_time : reference time for yearday calculation
#     @return:
#             float_df (pd.DataFrame): 
#     """
#     float_df = float_df.reset_index()
#     float_df.rename(columns={'LATITUDE':'latitude','LONGITUDE':'longitude', 'TIME':'datetime', 'CYCLE_NUMBER':'cycle_number',
#                             'PLATFORM_NUMBER':'wmoid', 'PRES':'pressure', 'TEMP':'temperature', 'PSAL':'salinity',
#                             'PRES_ERROR': 'pres_error', 'PSAL_ERROR': 'psal_error', 'TEMP_ERROR': 'temp_error',}, inplace=True)
#     float_df['yearday'] = myocn.datetime2ytd(float_df['datetime'], ref_time = ref_time)

#     # Create a unique profile id so easy to sort later
#     # Make sure strings are filled so 1st and 10th profile are different
#     prof = [tag.zfill(3) for tag in float_df['cycle_number'].astype(str)]
#     float_df['profid'] = [str(float_df.wmoid[0]) + '_id' + tag for tag in prof]

#     # Add calculated variables using gsw
#     float_df['SA']= gsw.SA_from_SP(float_df['salinity'],float_df['pressure'],float_df['longitude'],float_df['latitude'])
#     float_df['CT'] = gsw.CT_from_t(float_df['SA'], float_df['temperature'], float_df['pressure']) # Make sure to use SA here
#     float_df['sigma0'] = gsw.sigma0(float_df.SA.values, float_df.CT.values)
#     float_df['spice'] = gsw.spiciness0(float_df["SA"].values, float_df["CT"].values)

#     if var_list == 'phys':
#             var_list = ['wmoid', 'profid', 'latitude', 'longitude', 'datetime', 'yearday',
#             'pressure', 'CT', 'SA', 'sigma0', 'spice',
#             'temperature', 'salinity',
#             'temp_error', 'psal_error', 'pres_error', 'cycle_number']

#     return float_df[var_list]


def filter_qc_flags(float_df, qc_vars = 'all', use_flags=['1', '2', '5', '8']):
        """
        Filter a dataframe based on QC flags.
        Can choose different QC flags for different variables by calling the function multiple times.
        Note Argopy has this function, but this one allows you to track #obs, filter on position QC.
        @param: float_df (pd.DataFrame): dataframe of float data
                qc_vars (list): list of QC variables to filter
                        default 'all' filters on any variable with '_qc' in the name
                use_flags : flags that pass QC; default are standard argo QC flags 1, 2, and 8
                        '1' for 'good' data (only '1' returned in 'research' mode)
                        '2' for 'probably good' data
                        '5' for 'changed' data (rare; for position qc where lat/lon was adjusted)
                        '8' for 'interpolated/estimated' data
        @return: float_qc (pd.DataFrame)
        """ 
        print('Using flags: ', use_flags)
        float_qc = float_df.copy().reset_index()
        print ('# of profiles before QC filtering: \t', len(float_qc.profid.unique()))
        print('# of obs before QC filtering: \t\t', len(float_qc), '\n')

        if qc_vars == 'all':
                qc_vars = [var for var in float_qc.columns.tolist() if '_qc' in var]
        
        # for var in qc_vars:
        #         float_qc = float_qc[float_qc[var].isin(use_flags)]
        #         print('# of obs after ', var, ': \t\t', len(float_qc))
        

        qc_table = pd.DataFrame(columns= (use_flags + ['nobs_dropped', 'nobs_remaining']), index=qc_vars)
        for var in qc_vars:
                prevlen = len(float_qc) # store length before filtering
                for flag in use_flags:
                        qc_table.loc[var, flag] = len(float_qc[float_qc[var] == flag])

                # Filter based on use_flags
                float_qc = float_qc[float_qc[var].isin(use_flags)]
                qc_table.loc[var, 'nobs_dropped'] = int(prevlen - len(float_qc))
                qc_table.loc[var, 'nobs_remaining'] = len(float_qc)
        
        print(qc_table)

        print ('\n# of profiles before QC filtering: \t', len(float_qc.profid.unique()))
        return float_qc
        

def to_xr_dataset(argoDF, title, source):
    """ 
    Convert float Dataframe to Dataset, and assign title and source attributes.
    """
    argo_DS = xr.Dataset.from_dataframe(argoDF)
    argo_DS = argo_DS.set_coords([ 'wmoid', 'datetime', 'yearday', 'latitude', 'longitude'])
    argo_DS = argo_DS.assign_attrs({'title':title, 'source': source, 'date':str(datetime.datetime.now())})

    return argo_DS
    


# %% Interpolation function
# =============================================================================

def interpolate_float_pressure(argoDF, pres_levels = np.arange(0,1002,2), var_list = 'phys', ref_time = '2014-01-01'):
    ''' 
    Interpolate variables to a new pressure grid, without extrapolation by default.

    @ param  argoDF: dataframe with "profid" as a variable column
                      can hold profiles from one float, or from multiple floats
             pres_levels: (regular) pressure levels to interpolate to
             var_list: list of variables to interpolate, or 'phys' for core Argo variables
    @ return argoDF_regular: dataframe with interpolated variables
                            can be converted to xr Dataset with .to_xr_dataset()
    '''
    if var_list == 'phys':
        var_list = ['CT', 'SA', 'sigma0', 'spice', 'temperature', 
                    'salinity', 'yearday', 'latitude', 'longitude']

    # # Initialize a list of dataframes, one for each profid, with interpolated values
    profile_interp_list = []
    interp_dict_perProfile = {k:None for k in var_list}

    for profid, group in argoDF.groupby('profid'):
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

    argoDF_regular = pd.concat(profile_interp_list).dropna(subset=['CT', 'SA'])
    argoDF_regular = argoDF_regular.set_index(["profid", 'pressure'])
    argoDF_regular['datetime'] = pd.to_datetime(myocn.ytd2datetime(argoDF_regular['yearday'], ref_time = ref_time))

    # Make sure there are values for CT, SA
    argoDF_regular = argoDF_regular.dropna(subset=['CT', 'SA'])

    return argoDF_regular



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