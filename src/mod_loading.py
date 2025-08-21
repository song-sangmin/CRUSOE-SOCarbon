import xarray as xr
import pandas as pd
import numpy as np
import mod_ocean
import gsw
import datetime
# from mod_ocean import datetime2ytd


def import_data(type=['core', 'bgc', 'socat']):
    """ 
    Main import function for CRUSOE v2025
    # import mod_loading as loader
    # [coreDS, coreINDEX, bgcDS, bgcINDEX, socat] = loader.import_data(type=['core', 'bgc', 'socat'])
    """
    # # ADAPTING JUN 17 2025

    result = []

    # This is outside because we may need to match bgc to core even without returning
    filepath = '/Volumes/cremas-repo/data/core/L3-interp/'
    coreDS = xr.open_dataset(filepath + 'coreDATA_valid_interp_2014-2023_acc20250424.nc')
    coreINDEX = xr.open_dataset(filepath + 'coreINDEX_valid_interp_2014-2023_acc20250424.nc')
    if 'core' in type:
        result.append(coreDS)
        result.append(coreINDEX)

    if 'bgc' in type:
        filepath = '/Volumes/cremas-repo/data/bgc/L3-interp/'

        # Option 1 (default to this): All profiles with good pH above 25dbar. 
        bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_2014-2023_acc20250729.nc') # Updated; More pressure levels
        bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_2014-2023_acc20250729.nc')
        # bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_2014-2023_acc20250313.nc') # Older; ewer pressure levels
        # bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_2014-2023_acc20250313.nc')

        # Option 2: Include oxygen (not QC'ed) for diagnostic
        bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_noOxygenQC_2014-2023_acc20250729.nc') # Fewer pressure levels
        bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_noOxygenQC_2014-2023_acc20250729.nc')
        
        # Filter out any profiles without good Core files (only found from one float, 33 profiles without associated core data)
        missing_ids = np.setdiff1d(bgcINDEX.profid.values, coreINDEX.profid.values)
        bgcINDEX = bgcINDEX.sel(profid=~np.isin(bgcINDEX.profid, missing_ids), drop=True)
        bgcDS = bgcDS.sel(profid=~np.isin(bgcDS.profid, missing_ids), drop=True)

        # Make sure bgcINDEX has the same profids as bgcDS
        bgcINDEX = bgcINDEX.sel(profid=bgcDS.profid.values)
        result.append(bgcDS)
        result.append(bgcINDEX)


    if 'socat' in type:
        filepath = '/Volumes/cremas-repo/data/socat/L2-mask/' 
        # socat = pd.read_csv(filepath + 'SOCATv2024_SO_resampled_3h_median_acc20250121.csv')
        socat = xr.open_dataset(filepath + 'SOCATv2024_SO_3h_open_ocean_INDEX_acc20250314.nc')
        socat['yearday'] = mod_ocean.datetime2ytd(socat['datetime'].astype('datetime64[ns]'), ref_time='2014-01-01')
        socat = socat.where(socat.latitude<-35, drop=True)
        result.append(socat)

    return result # coreDS, coreINDEX, bgcDS, bgcINDEX, socat

def import_regresssion_data():
    # Call in classified coreDS and coreINDEX
    # Each profile associated with a class; all posterior probs given.
    filepath = '../working-vars/pcm/'

    bgcDS = xr.open_dataset(filepath + 'clustered_bgcArgo_output.nc')
    socatDS = xr.open_dataset(filepath + 'clustered_socat_output.nc')
    
    # Get bgcINDEX from above function
    [_, bgcINDEX] = import_data(type=['bgc'])
    # Drop any BGC profiles that were not classified
    missing_ids = np.setdiff1d(bgcINDEX.profid.values, bgcDS.profid.values)
    bgcINDEX = bgcINDEX.sel(profid=~np.isin(bgcINDEX.profid, missing_ids), drop=True)


    # # Prepare socatDS
    socatDS = socatDS.rename_vars({'index': 'sid'})  # Rename
    socatDS['sample_depth'] = socatDS['sample_depth'].fillna(0) # CHECK IF WE SHOULD DO THIS
    socatDS['pressure'] = gsw.p_from_z(socatDS.sample_depth, socatDS.latitude)

    #  Compute CT, SA
    socatDS['SA'] = gsw.SA_from_SP(socatDS.sal, socatDS.pressure, socatDS.longitude, socatDS.latitude)
    socatDS['CT'] = gsw.CT_from_t(socatDS.SA, socatDS.sst, socatDS.pressure)

    return [bgcINDEX, bgcDS, socatDS]

def main():
    print('Loading data...')

if __name__ == "__main__":
    main()