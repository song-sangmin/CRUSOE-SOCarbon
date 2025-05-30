# os tools


import numpy                 as np
import pandas                as pd
import xarray                as xr
import gsw
import scipy

# %% Interpolation function
# =============================================================================

def interpolate_float_pressure(argoDF, pres_levels = np.arange(0,1001,5), bgc_list = [], ref_time = '2014-01-01'):
    ''' 
    Interpolate variables to a new pressure grid, without extrapolation.

    @ param  argoDF: dataframe with "profid" as a variable column
                      can hold profiles from one float, or from multiple floats
             pres_levels: (regular) pressure levels to interpolate to
             bgc_list: (default) [] empty list for core Argo
                        ['pH'] or list of bgc variables to interpolate
    @ return argoDF_regular: dataframe with interpolated variables
                            can be converted to xr Dataset with .to_xr_dataset() in module
    '''
    # Default variables to interpolate (CT, SA) from core Argo 
    var_list = ['CT', 'SA', 'sigma0', 'spice', 'temperature', 
                    'salinity', 'yearday', 'latitude', 'longitude']
    var_list = var_list + bgc_list

    # # Initialize a list to hold interpolated DFs, one for each profile
    profile_interp_list = []

    # For each profile, interpolate variables and store in a dictionary
    interpolated_variables = {k:None for k in var_list} # For each 
    for profid, group in argoDF.groupby('profid'):
        
        # print('Processing profile:', profid)
        group = group.dropna(subset=['CT']).sort_values(by='pressure').reset_index()
        group = group.drop_duplicates(subset='pressure', keep="last") # why are there pressure duplicates? 
        for var in var_list:
            try: 
                f = scipy.interpolate.PchipInterpolator(x=group['pressure'], y=group[var], extrapolate = False)
                interpolated_variables[var] = f(pres_levels)
            except: 
                interpolated_variables[var] = np.tile(np.nan, len(pres_levels))
        
        # Combine profile information into one DF and add to list
        prof_interp = pd.DataFrame({var: interpolated_variables[var] for var in var_list})
        prof_interp['profid'] = profid
        prof_interp['wmoid'] = group['wmoid'] #.values[0]
        prof_interp['pressure'] = pres_levels

        profile_interp_list.append(prof_interp)

    argoDF_regular = pd.concat(profile_interp_list).dropna(subset=['CT', 'SA'])
    argoDF_regular = argoDF_regular.set_index(["profid", 'pressure'])
    argoDF_regular['datetime'] = pd.to_datetime(myocn.ytd2datetime(argoDF_regular['yearday'], ref_time = ref_time))

    # Make sure there are values for CT, SA
    argoDF_regular = argoDF_regular.dropna(subset=['CT', 'SA'])

    return argoDF_regular