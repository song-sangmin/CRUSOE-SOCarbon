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
    Files were created in 0.0 – 0.5
    # import mod_loading as loader
    # [coreDS, coreINDEX, bgcDS, bgcINDEX, socat] = loader.import_data(type=['core', 'bgc', 'socat'])
    """
    # # ADAPTING JUN 17 2025

    result = []

    # This is outside because we may need to match bgc to core even without returning
    filepath = '/Volumes/crusoe-repo/data/core/L3-interp/'
    coreDS = xr.open_dataset(filepath + 'coreDATA_valid_interp_2014-2023_acc20250424.nc')
    coreINDEX = xr.open_dataset(filepath + 'coreINDEX_valid_interp_2014-2023_acc20250424.nc')
    if 'core' in type:
        result.append(coreDS)
        result.append(coreINDEX)

    if 'bgc' in type:
        filepath = '/Volumes/crusoe-repo/data/bgc/L3-interp/'

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
        filepath = '/Volumes/crusoe-repo/data/socat/L2-mask/' 
        # socat = pd.read_csv(filepath + 'SOCATv2024_SO_resampled_3h_median_acc20250121.csv')
        # socat = xr.open_dataset(filepath + 'SOCATv2024_SO_3h_open_ocean_INDEX_acc20250314.nc')

        # new resample at 1d resolution jan 2 in 0.1_socat2024 processing
        socat = xr.open_dataset(filepath + 'SOCATv2024_SO_1d_open_ocean_INDEX_acc20260102.nc')

        socat['yearday'] = mod_ocean.datetime2ytd(socat['datetime'].astype('datetime64[ns]'), ref_time='2014-01-01')
        socat = socat.where(socat.latitude<-35, drop=True)
        result.append(socat)

    return result # coreDS, coreINDEX, bgcDS, bgcINDEX, socat

def import_clustering_results():
    """ 
    Files were created in 1.1_pcm_fit_coreArgo.ipynb
    [Y_gmm, allprobs, gmm_desc, coreDF] = loader.import_clustering_results()
    ~ 2 sec
    """
    gmm_desc = '8pc_8class_500dbar'
    Y_gmm = pd.read_csv('../working-vars/pcm/Y_gmm_20251201.csv', index_col=0) # Results of GMM

    datetag = '20251208'
    allprobs = pd.read_csv('../working-vars/pcm/postprobs_allclasses_' + datetag + '.csv')
    allprobs = allprobs.rename(columns = {str(k):(k+1) for k in range(8)})

    coreINDEX = import_data(type=['core'])[1]
    coreDF = coreINDEX.to_dataframe()
    coreDF['prof_datetag'] = coreDF.datetime.astype(str)
    coreDF['prof_datetag'] = coreDF['prof_datetag'].apply(lambda x: x.replace('-','').split(' ')[0])
    coreDF['prof_datetag'] = coreDF.apply(lambda row: str(int(row['wmoid'])) + '_' + str(row['prof_datetag']), axis=1)
    # coreDF

    return [Y_gmm, allprobs, gmm_desc, coreDF]

# used to be import_regression_data
def import_clustered_data():
    """ 
    Files were created in 1.2_pcm_classify_bgcArgo_ship.ipynb
    Colocation needed to cluster SOCAT data was done in 0.5_socat2024_colocation.ipynb 
    """
    # Call in classified coreDS and coreINDEX
    # Each profile associated with a class; all posterior probs given.
    filepath = '../working-vars/pcm/'

    gmm_desc = '8pc_8class_500dbar'
    bgcDS = xr.open_dataset(filepath + 'clustered_bgcArgo_acc20251201.nc')

    # updated to 1day resample, jan 2026
    socatDS = xr.open_dataset(filepath + 'clustered_socat_1d_acc20260111.nc')
    
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

def import_regression_inputs(addSatelliteFeatures = False):
    """ 
    Files were created in 2.0_process_regression_data.ipynb
    These have all variables for regression, including target pco2
    Used as inputs in notebooks 2.1 – 
    
    SOCAT fco2 has been converted to pco2 using NCEP atmospheric pressures
    """

    # These have pco2 and delta-pco2 (ocean minus atmosphere, in uatm)
    filepath = '../working-vars/regression/inputs/'
    float_parameter = 'pHbias5_pK1'
    datetag = '20260119' #20251209'
    soccomDF = pd.read_csv(filepath + 'floatDF_soccom20m_pco2_' + float_parameter + '_PCM_8pc8class_acc' + datetag + '.csv',
    index_col=0)

    datetag = '20260111' #'20260111'
    socatDF = pd.read_csv(filepath + 'shipDF_socatv2024_1d_pCO2converted_co2sys_PCM_8pc8class_acc' + datetag + '.csv',
    index_col=0)

    if addSatelliteFeatures:
        # return output of 2.1_add_regression_features.ipynb
        print('Using data with added satellite features (ADT, SLA)')
        # datetag = '20260112'
        # soccomDF = pd.read_csv(filepath + 'floatDF_ADT_SLA_soccom20m_pco2_pHbias3_pK1_PCM_8pc8class_acc' + datetag + '.csv', 
        #                     index_col=0)
        datetag = '20260119'
        soccomDF = pd.read_csv(filepath + 'floatDF_ADT_SLA_soccom20m_pco2_pHbias5_pK1_PCM_8pc8class_acc' + datetag + '.csv', 
                            index_col=0)
        datetag = '20260112'
        socatDF = pd.read_csv(filepath + 'shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_PCM_8pc8class_acc' + datetag + '.csv', 
                              index_col=0)

    return [soccomDF, socatDF]

def import_training_validation(platform='combined'):
    """ 
    Files were created in 2.2_subset_train_validation.ipynb
    platform: 'float', 'ship', 'combined'
    """
    filepath = '../working-vars/regression/inputs/'
    datetag = '20260112' 

    if platform=='float':
        trainDF = pd.read_csv(filepath + 'mldata_trainingDF_float_8PC8class_acc' + datetag + '.csv', index_col=0)
        valDF = pd.read_csv(filepath + 'mldata_validationDF_float_8PC8class_acc' + datetag + '.csv', index_col=0)
    elif platform=='ship':
        trainDF = pd.read_csv(filepath + 'mldata_trainingDF_ship_8PC8class_acc' + datetag + '.csv', index_col=0)
        valDF = pd.read_csv(filepath + 'mldata_validationDF_ship_8PC8class_acc' + datetag + '.csv', index_col=0)
    else: # combined
        trainDF = pd.read_csv(filepath + 'mldata_trainingDF_combinedShipFloat_8PC8class_acc' + datetag + '.csv', index_col=0)
        valDF = pd.read_csv(filepath + 'mldata_validationDF_combinedShipFloat_8PC8class_acc' + datetag + '.csv', index_col=0)

    return [trainDF, valDF]


def main():
    print('Loading data...')

if __name__ == "__main__":
    main()