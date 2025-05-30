# Module for accessing Argo data with Argopy
# 
# Song Sangmin 
# sangsong@uw.edu
# Apr 2025
# ===========

import os.path
from tqdm import tqdm

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
        # all_wmoids = pd.DataFrame(argoDS.wmoid)[0].unique()
        all_wmoids = argoDS.to_dataframe().wmoid.unique()
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



# Main method
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

        print ('\n# of profiles after QC filtering: \t', str(len(float_qc.profid.unique())) + '\n')
        return float_qc
        

def to_xr_dataset(argoDF, nc_attrs={'date':str(datetime.datetime.now())}):
    """ 
    Convert float Dataframe to Dataset, and assign title and source attributes.
    """
    temp = xr.Dataset.from_dataframe(argoDF)
    # argo_INDEX = temp.mean(dim='pressure')

    argo_DS = temp.set_coords([ 'wmoid', 'datetime', 'yearday', 'latitude', 'longitude'])
    argo_DS = argo_DS.assign_attrs(nc_attrs)

    return argo_DS
    


# %% Interpolation functions
# =============================================================================
def get_max_gap(x_val):
    """
    Return the maximum allowed gap based on the value of x.
    """
    if x_val < 100:
        return 25
    elif x_val < 500:
        return 50
    elif x_val < 1000:
        return 100
    else:
        return 250
    

def interpolate_z_profile(prof, pres_levels, var_list, 
                          z_gap = 200, surface_fill = 0,
                          ref_time = '2014-01-01') -> pd.DataFrame:
    """  
    Function to interpolate single float profile data (pchip) to chosen pressure levels,
    Does not fit over vertical gaps in the data, with cutoff defined by z_gap. 

    @param      prof: pd DataFrame with data from a single profile
                pres_levels: list of pressure levels to interpolate to
                var_list: list of variables to interpolate
                z_gap: z (m) threshold for large gaps in the data, can define own dbar
                    or use 'dynamic' to use get_max_gap() function
                surface_fill: (dbar) pressure above which to fill surface values with nearest neighbor,
                                 if first observed value is shallower than surface_fill
    @return     output: pd DataFrame with interpolated data
    """

    # If there is data, separate profile out into continuous vertical parts
    # Useful catch when iterating over a lot of profiles using regrid_pressure_levels() 
    try:
        # Sort by pressure and drop duplicates
        prof = prof.reset_index().sort_values(by='pressure')
        prof = prof.drop_duplicates(subset='pressure', keep="last") 
        prof['pres_diff'] = [np.nan] + np.diff(prof.pressure).tolist()

        # Dynamic gap filling
        if z_gap == 'dynamic':
            prof['max_allowed_gap'] = ([get_max_gap(x) for x in prof['pressure'].values])
            subID_index = prof[prof['pres_diff'] > prof['max_allowed_gap']].index
        else: # Static, define own
            subID_index = prof[prof['pres_diff'] > z_gap].index

        # print(subID_index)
        # Separate out continuous ID's
        prof.loc[subID_index, 'marker'] = 1
        prof['continuous_id'] = prof['marker'].cumsum().ffill().fillna(0).astype(int)
    except: 
        # print('no data in profid' + str(prof.profid.iloc[0]))
        return None
    
    # Initialize interpolated DataFrame to return, with pressure as index
    output = pd.DataFrame(index=pres_levels, columns=var_list, dtype=float)
    output.index.name = 'pressure'

    # If there are no gaps, all points will be assigned to same continuous_id
    # Fit pchip to each continuous_id 
    for _, subprof in prof.groupby('continuous_id'):
        xmin = subprof.pressure.min(); xmax = subprof.pressure.max()
        subpres_levels = [x for x in pres_levels if x > xmin and x < xmax] # valid pres levels
        if len(subprof) > 1: # If there is more than 1 point to fit pchip over
            for var in var_list:
                subprof = subprof.dropna(subset=[var]) # y must have finite values
                f = scipy.interpolate.PchipInterpolator(x=subprof['pressure'], y=subprof[var], extrapolate = False)
                output.loc[subpres_levels, var] = f(subpres_levels)

    if surface_fill > 0:
        # # ==== HANDLE SURFACE GAPS 
        # Fill surface with the nearest neighbor if the first observed value is below x_fill = 25
        first_obs = prof.iloc[0]
        if first_obs['pressure'] < surface_fill:
            for var in var_list:
                # Decide here if you want to fill with first interpolated value, or first observed value
                # fill_value = output.dropna().iloc[0].y_interp # first interp
                fill_value = prof.iloc[0][var] # first observed

                fill_index = [x for x in output.index if x < first_obs.pressure]
                output.loc[fill_index, var] = np.tile(fill_value, len(fill_index))

    # Reset index, drop nans (needed before datetime calculation)
    output = output.reset_index().dropna() 

    # Add profid, wmoid, datetime
    output['profid'] = np.tile(prof.profid.iloc[0], len(output))
    output['wmoid'] = np.tile(prof.wmoid.iloc[0], len(output))
    
    if 'yearday' in var_list:
        output['datetime'] = pd.to_datetime(myocn.ytd2datetime(output.yearday, ref_time = ref_time))
    
    return output
    

def regrid_pressure_levels(argoDF, pres_levels = np.arange(0,1001,5), var_list = 'all', bgc_list = [], 
                           z_gap = 200, surface_fill =0, ref_time = '2014-01-01'):
    ''' 
    Interpolate variables to a new pressure grid, without extrapolation.
    
    Replaces old function 'interpolate_float_pressure' from mod_argo in 0.4_bgcArgo 
    Adding middle catch for large gaps in the data

    @ param  argoDF: dataframe with "profid" as a variable column
                      can hold profiles from one float, or from multiple floats
             pres_levels: pressure levels to interpolate to
             bgc_list: (default) [] empty list for core Argo
                        ['pH'] or list of bgc variables to interpolate
                z_gap: z (m) threshold for large gaps in the data, can define own dbar
                        Set as very large value if you want to interpolate smoothly over gaps
                surface_fill: fill surface with nearest neighbor if first observed value is below x_fill
                            see interpolate_z_profile() for details
                ref_time: reference time for yearday calculation
    @ return argoDF_regular: dataframe with interpolated variables
                            can be converted to xr Dataset with .to_xr_dataset()
    '''
    # Default variables to interpolate (CT, SA) from core Argo 
    if var_list == 'all':
        var_list = ['CT', 'SA', 'sigma0', 'spice', 'temperature', 
                        'salinity', 'yearday', 'latitude', 'longitude']
    var_list = var_list + bgc_list

    # Initialize list to hold interpolated DFs, one for each profile
    profile_interp_list = []
    for key, profDF in argoDF.groupby('profid'):
        profile_interp_list.append(interpolate_z_profile(profDF, pres_levels, var_list, 
                                            z_gap = z_gap,  # avoid interpolating over large gaps 
                                            surface_fill = surface_fill,
                                            ref_time = ref_time))

    # Combine all profiles into one DataFrame
    argoDF_regular = pd.concat(profile_interp_list) 
    argoDF_regular = argoDF_regular.set_index(["profid", 'pressure'])
    # argoDF_regular = argoDF_regular.dropna(subset=['CT', 'SA'])

    return argoDF_regular
