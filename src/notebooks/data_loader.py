# Module for loading raw and processed data for all notebooks
# previously mod_loading.py, moved out of src/crusoe/ to src/notebooks/

import xarray as xr
import pandas as pd
import numpy as np
from crusoe import mod_ocean
import gsw
import datetime
from datetime import datetime, timedelta
import os
import scipy.io as sio
# from mod_ocean import datetime2ytd
from crusoe import mod_preprocessing as mod_prep

# [coreDS, coreINDEX] = loader.import_core_data(type = 'L3_only')
# [bgcDS, bgcINDEX] = loader.import_bgc_data(type = 'L3_only')


def import_core_data(type='L3_only'):
    """ 
    Main import function for CRUSOE v2025
    Files were created in 0.0 – 0.5
    [coreDS, coreINDEX] = loader.import_core_data(type = 'L3_only')

    # OLD:  [coreDS, coreINDEX, bgcDS, bgcINDEX, socat] = loader.import_data(type=['core', 'bgc', 'socat'])
    """
    filepath = '/Volumes/crusoe-repo/data/core/L3-interp/'
    coreINDEX = xr.open_dataset(filepath + 'coreINDEX_valid_interp_2014-2023_acc20250424.nc')
   

    if type == 'L3_only': # original data used before clustering
        coreDS = xr.open_dataset(filepath + 'coreDATA_valid_interp_2014-2023_acc20250424.nc')

    elif type == 'L3_extended': # original ten-year plus 2024 data (L3_only + 2024)
        # Extended record to 2024, for clustering SOCAT 2024 test data
        coreINDEX = xr.open_dataset(filepath + 'coreINDEX_valid_interp_2014-2024_acc20260325.nc')
        # coreDS = xr.open_dataset(filepath + 'coreDATA_valid_interp_2014-2024_acc20260325.nc')
        coreDS = xr.open_dataset(filepath + '/coreDATA_valid_interp_MLD_2014-2024_acc20260325.nc')
    
    elif type == '2024': # for clustering, so that socat2024 test data can be used
        # includes mld
        coreINDEX = xr.open_dataset(filepath + 'coreINDEX_test_valid_interp_MLD_2024only_acc20260327.nc')
        coreDS = xr.open_dataset(filepath + 'coreDATA_test_valid_interp_MLD_2024only_acc20260327.nc')

    elif type == 'processed': # # with MLD and ADT added
        # BEST VERSION AS OF FEB 6
        # These are in xr Dataset format
        folder = '../working-vars/L4-datasets/'
        coreINDEX = xr.open_dataset(folder + 'coreINDEX_processed_2014-2023_atmosco2_mld_adt_acc20260211.nc')
        coreDS = xr.open_dataset(folder + 'coreDATA_processed_2014-2023_mld_adt_sla_acc20260211.nc')
        #coreINDEX is used for application, dataframe version is in regression/inputs/P1-processed/
                                    
    return [coreDS, coreINDEX]


def import_bgc_data(type = 'L3_only'):
    """ 
    [bgcDS, bgcINDEX] = loader.import_bgc_data(type = 'L3_only')
    """
    filepath = '/Volumes/crusoe-repo/data/bgc/L3-interp/'

    [coreDS, coreINDEX] = import_core_data(type = 'L3_only')

    if type == 'L3_only':
        # Option 1 (default to this): All profiles with good pH above 25dbar. 
        bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_2014-2023_acc20250729.nc') # Updated; More pressure levels
        bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_2014-2023_acc20250729.nc')
        # bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_2014-2023_acc20250313.nc') # Older; ewer pressure levels
        # bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_2014-2023_acc20250313.nc')

        # Option 2: Include oxygen (not QC'ed) for diagnostic
        # bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_noOxygenQC_2014-2023_acc20250729.nc') # Fewer pressure levels
        # bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_noOxygenQC_2014-2023_acc20250729.nc')
    
        # Filter out any profiles without good Core files (only found from one float, 33 profiles without associated core data)
        # print(len(bgcINDEX.profid.values))
        missing_ids = np.setdiff1d(bgcINDEX.profid.values, coreINDEX.profid.values)
        bgcINDEX = bgcINDEX.sel(profid=~np.isin(bgcINDEX.profid, missing_ids), drop=True)
        bgcDS = bgcDS.sel(profid=~np.isin(bgcDS.profid, missing_ids), drop=True)

        # Make sure bgcINDEX has the same profids as bgcDS
        bgcINDEX = bgcINDEX.sel(profid=bgcDS.profid.values)
        
    elif type == 'processed': # with clustering and MLD added
        folder = '../working-vars/argo/import-L3/'
        # datetag = '20260121'
        # bgcINDEX = xr.open_dataset(folder + 'bgcINDEX_validL3_2014-2023_withMLD_acc' + datetag + '.nc')
        bgcDS = xr.open_dataset(folder + 'bgcDATA_validL3_2014-2023_withMLD_acc' + '20260121' + '.nc')

        folder = '../working-vars/L4-datasets/' # new version with full co2, mld, adt
        bgcINDEX = xr.open_dataset(folder + 'bgcINDEX_processed_2014-2023_fullco2_mld_adt_acc20260211.nc')
    
    elif type == 'test': # 2024 data only. for dataframe version, use import_ML_dataframes():
        # folder = '/Volumes/crusoe-repo/data/bgc/L3-interp/'
        # bgcDS = xr.open_dataset(folder + 'bgcDATA_valid_interp_2024_TESTING_acc' + '20260209' + '.nc')
        # # Make sure bgcDS profids don't overlap with coreDS (numbering started over for test period)
        # temp = bgcDS.to_dataframe().reset_index().copy()
        # temp['profid'] = temp['profid'].apply(lambda x: x.replace('id', 'testid'))
        # bgcDS = temp.set_index(['profid', 'pressure']).to_xarray()


        folder = '../working-vars/L4-datasets/' # new version with full co2, mld, adt
        bgcINDEX = xr.open_dataset(folder + 'testINDEX_processed_2024_fullco2_mld_adt_acc20260211.nc')
        bgcDS = xr.open_dataset(folder + 'testDS_processed_2024_fullco2_mld_adt_acc20260211.nc')

    return [bgcDS, bgcINDEX]


# def import_float_20m_data():


def import_socat_L2(version='v2024'):
    # Masked to open ocean south of 35S
    filepath = '/Volumes/crusoe-repo/data/socat/L2-mask/' 

    # new resample at 1d resolution jan 2 in 0.1_socat2024 processing
    if version == 'v2024':
        socat = xr.open_dataset(filepath + 'SOCATv2024_SO_1d_open_ocean_INDEX_acc20260102.nc')
    else: # 'v2025'
        # socat = xr.open_dataset(filepath + 'SOCATv2025_SO_1d_open_ocean_INDEX_acc20260324.nc') # includes socat 2024 test data
        socat = xr.open_dataset(filepath + 'SOCATv2025_SO_1d_open_ocean_INDEX_acc20260518.nc') # 23.5h resample instead of 24 

    # change to linear_time jan 21 2026
    # socat['yearday'] = mod_ocean.datetime2ytd(socat['datetime'].astype('datetime64[ns]'), ref_time='2014-01-01')
    socat['linear_time'] = mod_ocean.datetime2ytd(socat['datetime'].astype('datetime64[ns]'), ref_time='2014-01-01')
    socat = socat.where(socat.latitude<-35, drop=True)

    return socat # coreDS, coreINDEX, bgcDS, bgcINDEX, socat


# def import_data(type=['core', 'bgc', 'socat']):
#     """ 
#     Main import function for CRUSOE v2025
#     Files were created in 0.0 – 0.5
#     # import mod_loading as loader
#     # [coreDS, coreINDEX, bgcDS, bgcINDEX, socat] = loader.import_data(type=['core', 'bgc', 'socat'])
#     """
#     # # ADAPTING JUN 17 2025

#     result = []

#     # This is outside because we may need to match bgc to core even without returning
#     filepath = '/Volumes/crusoe-repo/data/core/L3-interp/'
#     coreDS = xr.open_dataset(filepath + 'coreDATA_valid_interp_2014-2023_acc20250424.nc')
#     coreINDEX = xr.open_dataset(filepath + 'coreINDEX_valid_interp_2014-2023_acc20250424.nc')
#     if 'core' in type:
#         result.append(coreDS)
#         result.append(coreINDEX)

#     if 'bgc' in type:
#         filepath = '/Volumes/crusoe-repo/data/bgc/L3-interp/'

#         # Option 1 (default to this): All profiles with good pH above 25dbar. 
#         bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_2014-2023_acc20250729.nc') # Updated; More pressure levels
#         bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_2014-2023_acc20250729.nc')
#         # bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_2014-2023_acc20250313.nc') # Older; ewer pressure levels
#         # bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_2014-2023_acc20250313.nc')

#         # Option 2: Include oxygen (not QC'ed) for diagnostic
#         bgcDS = xr.open_dataset(filepath + 'bgcDATA_valid_interp_noOxygenQC_2014-2023_acc20250729.nc') # Fewer pressure levels
#         bgcINDEX = xr.open_dataset(filepath + 'bgcINDEX_valid_interp_noOxygenQC_2014-2023_acc20250729.nc')
        
#         # Filter out any profiles without good Core files (only found from one float, 33 profiles without associated core data)
#         missing_ids = np.setdiff1d(bgcINDEX.profid.values, coreINDEX.profid.values)
#         bgcINDEX = bgcINDEX.sel(profid=~np.isin(bgcINDEX.profid, missing_ids), drop=True)
#         bgcDS = bgcDS.sel(profid=~np.isin(bgcDS.profid, missing_ids), drop=True)

#         # Make sure bgcINDEX has the same profids as bgcDS
#         bgcINDEX = bgcINDEX.sel(profid=bgcDS.profid.values)
#         result.append(bgcDS)
#         result.append(bgcINDEX)


#     if 'socat' in type:
#         filepath = '/Volumes/crusoe-repo/data/socat/L2-mask/' 
#         # socat = pd.read_csv(filepath + 'SOCATv2024_SO_resampled_3h_median_acc20250121.csv')
#         # socat = xr.open_dataset(filepath + 'SOCATv2024_SO_3h_open_ocean_INDEX_acc20250314.nc')

#         # new resample at 1d resolution jan 2 in 0.1_socat2024 processing
#         socat = xr.open_dataset(filepath + 'SOCATv2024_SO_1d_open_ocean_INDEX_acc20260102.nc')

#         socat['yearday'] = mod_ocean.datetime2ytd(socat['datetime'].astype('datetime64[ns]'), ref_time='2014-01-01')
#         socat = socat.where(socat.latitude<-35, drop=True)
#         result.append(socat)

#     return result # coreDS, coreINDEX, bgcDS, bgcINDEX, socat

def import_socat_colocation(buffer_time = '7d'):
    """ 
    # may 2026
    Resampled at 23.5h instead of 24h in 0.1_socat_processing
   
    # mar 2026 '20260327'
    Now using 1.2_processing
    Colocate to coreindex (2014-2024) using v2025
    

    # jan 2026 (20260111)
    Used for classification in 1.2_pcm_classify_bgcArgo_ship.ipynb
    Files were created in 0.5_socat2024_colocation.ipynb
    """
    # filepath = '../working-vars/socat/colocate-coreArgo/'
    filepath = '/Volumes/crusoe-repo/data/socat/colocate-coreArgo/' 
    sepdict_7d = {key:None for key in [str(x) for x in range(2014,2025)]}

    for x in os.listdir(filepath):
        if x.startswith('colocate_v2025_validArgo_7d') & x.endswith('20260518.csv'):
            # sepdict_7d[x[14:18]] = pd.read_csv(filepath+x, index_col=0)
            sepdict_7d[x[30:34]] = pd.read_csv(filepath+x, index_col=0)
            print('Imported data for _7d window: ' + x)

    sepstat_7d = pd.concat(sepdict_7d.values()).reset_index().drop(['level_0', 'index'], axis=1)

    return sepstat_7d

# %% PROCESSED DATAFRAMES /working-vars/regression/P1-processed/
def import_p1_processed():
    """ 
    Output of 2.0_RUN_preprocessing
    Returns dataframes that have mld, adt, atmospheric co2 added
    Not yet clustered
    """
    print('Importing processed dataframes...')
    folder = '../working-vars/regression/P1-processed/'

    # Version with mld, adt, but not sea ice or wind speed
    datetag = '20260211'
    # bgcArgo_trainval = pd.read_csv(folder + 'bgcArgo_trainval_processed_co2_mld_adt_soccom20m_pCO2_pHbias5_pK1_yr2014-2023_acc' + datetag + '.csv', index_col=0)
    # bgcArgo_test = pd.read_csv(folder + 'bgcArgo_test_processed_co2_mld_adt_soccom20m_pCO2_pHbias5_pK1_yr2024_acc' + datetag + '.csv', index_col=0)
    # socat_trainval = pd.read_csv(folder + 'socat_trainval_processed_co2_mld_adt_yr2014-2023_acc' + datetag + '.csv', index_col=0)
    coreArgo_application = pd.read_csv(folder + 'coreArgo_application_processed_co2_mld_adt_yr2014-2023_acc' + datetag + '.csv', index_col=0)

    # Newer version with sea icea nd wind speed, march 2026
    # may 2026 resampled socat at 23.5h 
    socat_test = pd.read_csv(folder + 'socatv2025_test_processed_co2_yr2024_acc20260518.csv', index_col=0)
    socat_trainval = pd.read_csv(folder + 'socatv2025_trainval_processed_co2_yr2014-2023_acc20260518.csv', index_col=0)
    bgcArgo_trainval = pd.read_csv(folder + 'bgcArgo_trainval_processed_co2_soccom20m_pCO2_pHbias5_pK1_yr2014-2023_acc20260327.csv', index_col=0)
    bgcArgo_test = pd.read_csv(folder + 'bgcArgo_test_processed_co2_soccom20m_pCO2_pHbias5_pK1_yr2024_acc20260327.csv', index_col=0)

    ### CHANGE THIS TO UPDATED FILE NAME FROM 1.0_RUN
    # coreArgo_application = pd.read_csv('../working-vars/regression/P1-processed/coreArgo_2014-2024_temporary_icefraction.csv', index_col=0)



    output = [bgcArgo_trainval, bgcArgo_test, socat_trainval, socat_test, coreArgo_application]
    for df in output:
        df['datetime'] = pd.to_datetime(df['datetime'])

    print('Returned [bgcArgo_trainval(2014-2023), bgcArgo_test(2024), ' \
                    'socat_trainval(2014-2023), socat_test(2024), coreArgo_application(2014-2023)]')
    return output # [bgcArgo_trainval, bgcArgo_test, socat_trainval, coreArgo_application]

# %% CLUSTERING

def import_p2_clustered(type = 'pcm_probs', pcm_params='pc8_gmm6', dbar_limit = 501, datetag = '20260327'):
    """ 
    Updated clustering after preprocessing Feb 11 2026

    @return         PCM_finder: dataframe with profid, class probabilities, assignment, prof_datetag
    """
    filepath = '../working-vars/regression/P2-clustered/' + pcm_params + '/'

    if type == 'pcm_probs':
        # outut from 2.1_PCM
        print('Importing clustering results for ' + pcm_params + '_' + str(dbar_limit) + '_dbar ' + '...')
        # if pcm_params == 'pc8_gmm6': datetag = '20260211' # feb 2026
        # if pcm_params == 'pc8_gmm6': datetag = '20260327' # march 2026. newer version with 2024 core included

        PCM_components = pd.read_csv(filepath + 'Y_gmm_' + str(dbar_limit) + 'dbar_' + pcm_params + '_' + datetag + '.csv', index_col=0) # Results of GMM
        PCM_finder = pd.read_csv(filepath + 'PCM_finder_' + str(dbar_limit) + 'dbar_' + pcm_params + '_' + datetag + '.csv', index_col=0) # Results of GMM
        # allprobs = pd.read_csv(filepath + 'postprobs_' + pcm_params + '_' + datetag + '.csv')  # already reindexed at 1

        # comment on for feb 2026
        if datetag == '20260211':
            PCM_finder = PCM_finder.set_index('profid')
        print('Returned [PCM_components, PCM_finder (probabilities)]')

        output = [PCM_components, PCM_finder]

    elif type == 'class-gapfilled': #
        # use datetag 0211 for before adding sea ice and adt
        #output from 2.2_PCM
        print('Importing processed Dataframes with gap-filled classes; ' + pcm_params + '...')
        # datetag = '20260327'
        bgcArgo_trainval = pd.read_csv(filepath + 'P2_bgcArgo-trainvalDF_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv', index_col=0)
        bgcArgo_test = pd.read_csv(filepath + 'P2_bgcArgo-testDF_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv', index_col=0)

        # datetag 0518 for resampled socat at 23.5h
        socat_trainval = pd.read_csv(filepath + 'P2_socat-trainvalDF_class-gapfilled_' + pcm_params + '_acc' + '20260518' + '.csv', index_col=0)
        socat_test = pd.read_csv(filepath + 'P2_socat-testDF_class-gapfilled_' + pcm_params + '_acc' + '20260518' + '.csv', index_col=0)
       
        # Split up socat_all (used to only be until 2023, switching to v2025 in mar 2026)
        # socat_all['datetime'] = pd.to_datetime(socat_all['datetime'])
        # socat_test = socat_all[socat_all['datetime'].dt.year == 2024].copy() # for now, use trainval with 2024 test data. will need to update when new version with sea ice and adt is ready
        # socat_trainval = socat_all[socat_all['datetime'].dt.year < 2024].copy()
        
        # datetag = '20260211'
        coreArgo_application = pd.read_csv(filepath + 'P2_coreArgo-applicationDF_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv', index_col=0)
        # bgcArgo_trainval = pd.read_csv(filepath + 'P2_bgcArgo-trainvalDF_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv', index_col=0)
        # bgcArgo_test = pd.read_csv(filepath + 'P2_bgcArgo-testDF_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv', index_col=0)
        # socat_trainval = pd.read_csv(filepath + 'P2_socat-trainvalDF_class-gapfilled_' + pcm_params + '_acc' + datetag + '.csv', index_col=0)
        # socat_test = pd.DataFrame()

       # The bgcArgo core profiles are still in coreArgo_application, but need to exclude ones that have associated bgc obs
        shared_profids = set(bgcArgo_trainval.index).intersection(set(coreArgo_application.index))
        core_only_profids = set(coreArgo_application.index) - set(shared_profids) # profids without associated 
        coreArgo_application = coreArgo_application.loc[[*core_only_profids]]

        output = [bgcArgo_trainval, bgcArgo_test, socat_trainval, socat_test, coreArgo_application]
        
        # for df in output:
        #     df['datetime'] = pd.to_datetime(df['datetime'])
            # df.loc[df.sea_ice.isna(), 'sea_ice'] = 0
            # print('Filled nan sea ice with 0s.')

    return output 


def import_p3_trainval(pcm_desc = 'pc8_gmm6_excludeClass5'):
    """ same as P2 but with any excluded classes removed 
    """
    folder = '../working-vars/regression/P3-trainval/'
    trainvalDF_all = pd.read_csv(folder + 'P3-trainval_bgcArgo_SOCAT_2014-2023_' + 
                                 'floatparam_pHbias5_pK1_' + pcm_desc + 
                                 '_acc20260211.csv')
    coreDF = pd.read_csv(folder + 'coreArgo_application_final_floatparam_pHbias5_pK1_' + pcm_desc + '_20260211.csv')
    testDF = pd.read_csv(folder + 'bgcArgo_test_final_floatparam_pHbias5_pK1_' + pcm_desc + '_20260211.csv')

    print(len(trainvalDF_all), 'profiles in trainvalDF_all (2014-2023); ', len(coreDF), 'profiles in coreDF (2014-2023)')
    print(len(testDF), 'profiles in testDF (2024 bgcArgo)')
    return [trainvalDF_all, coreDF, testDF]


def import_p4_pointobs(pcm_desc = 'pc8_gmm6_excludeClass5'):
    # Combined point obs for socat and bgc-argo (pco2_ocean, observed) and core estimates (pco2_ocean_pred)
    folder = '../working-vars/regression/P3-estimated/'
    filename = 'P3-point_pCO2_combined_core-bgc-socat_2014-2023_featD_pc8_gmm6_exclClass5_acc20260211.csv'

    pointDF = pd.read_csv(folder + filename, index_col=0)
    return pointDF

# def import_clustering_results(pcm_params='pc8_gmm8'):
#     """ 
#     originally used before running preprocessing in 2.0
#     as of feb 11 changing to make P1-processed datasets and dataframes first, then clustering
#     replace with above method 

#     Files were created in 1.1_pcm_fit_coreArgo.ipynb
#     [Y_gmm, allprobs, gmm_desc, coreDF_ave] = loader.import_clustering_results()
#     ~ 2 sec
#     """
#     filepath = '../working-vars/pcm/' + pcm_params + '/'

#     if pcm_params == 'pc8_gmm8':
#         # from dec 2025
#         Y_gmm = pd.read_csv(filepath + 'Y_gmm_20251201.csv', index_col=0) # Results of GMM
#         datetag = '20251208'
#         allprobs = pd.read_csv(filepath + 'postprobs_allclasses_' + datetag + '.csv')
#         allprobs = allprobs.rename(columns = {str(k):(k+1) for k in range(8)})
#     elif pcm_params == 'pc8_gmm6':
#         # updated jan 20226
#         datetag = '20260119'
#         Y_gmm = pd.read_csv(filepath + 'Y_gmm_501dbar_' + pcm_params + '_' + datetag + '.csv',
#                              index_col=0) # Results of GMM
#         allprobs = pd.read_csv(filepath + 'postprobs_allclasses_' + datetag + '.csv')
#         allprobs = allprobs.rename(columns = {str(k):(k+1) for k in range(8)})

#     elif pcm_params == 'pc8_gmm7':
#         datetag = '20260201'
#         Y_gmm = pd.read_csv(filepath + 'Y_gmm_501dbar_' + pcm_params + '_' + datetag + '.csv',
#                              index_col=0) # Results of GMM
#         allprobs = pd.read_csv(filepath + 'postprobs_allclasses_' + datetag + '.csv')  # already renamed
        
#     coreINDEX = import_core_data(type='L3_only')[1]

#     coreDF_ave = coreINDEX.to_dataframe()
#     coreDF_ave['prof_datetag'] = coreDF_ave.datetime.astype(str)
#     coreDF_ave['prof_datetag'] = coreDF_ave['prof_datetag'].apply(lambda x: x.replace('-','').split(' ')[0])
#     coreDF_ave['prof_datetag'] = coreDF_ave.apply(lambda row: str(int(row['wmoid'])) + '_' + str(row['prof_datetag']), axis=1)

#     return [Y_gmm, allprobs, coreDF_ave]

# used to be import_regression_data
def import_clustered_data(pcm_params='pc8_gmm8'):
    """ 
    


    Files were created in 1.2_pcm_classify_bgcArgo_ship.ipynb
    Colocation needed to cluster SOCAT data was done in 0.5_socat2024_colocation.ipynb 
    """
    # Call in classified coreDS and coreINDEX
    # Each profile associated with a class; all posterior probs given.
    filepath = '../working-vars/pcm/' + pcm_params + '/'

    if pcm_params == 'pc8_gmm8':
        bgcDS = xr.open_dataset(filepath + 'clustered_bgcArgo_acc20251201.nc')
        # updated to 1day resample, jan 2026
        socatDS = xr.open_dataset(filepath + 'clustered_socat_1d_acc20260111.nc')
        
    elif pcm_params == 'pc8_gmm7':
        datetag = '20260204'
        bgcDS = xr.open_dataset(filepath + 'clustered_bgcArgo_acc' + datetag + '.nc')
        socatDS = xr.open_dataset(filepath + 'clustered_socat_1d_acc' + datetag + '.nc')
    elif pcm_params == 'pc8_gmm6':
        datetag = '20260120'
        bgcDS = xr.open_dataset(filepath + 'clustered_bgcArgo_acc' + datetag + '.nc')
        socatDS = xr.open_dataset(filepath + 'clustered_socat_1d_acc' + datetag + '.nc')

        coreINDEX = xr.open_dataset(filepath + 'coreINDEX_classified_' + '20260119' + '.nc')
        core_classes = {k:dat for k, dat in coreINDEX.groupby('class')}

        
    # Get bgcINDEX from above function
    [_, bgcINDEX] = import_bgc_data()
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

    return [bgcINDEX, bgcDS, socatDS] #, coreINDEX]




def import_processed_inputs(float_param = 'pHbias5_pK1'):
    """ 
    Has training/validation data : bgcArgo 2014-2023, socat 2014-2023
    Test data: bgcArgo_2024 
    Application data: coreArgo 2014-2024


    with MLD and ADT added. updated feb 10 2026
    Files created in 2.0_RUN, originally after clustering
    Saving output into new foloder processed-< > so MLD 
    doesn't get recalculated every time with new clustering
    """
    floatDF = import_soccom_20m_averages(processed=True)

    # shipDF = pd.read_csv('../working-vars/socat/processed-socat2024/shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_acc20260203.csv', index_col=0)
    # shipDF['cluster'] = np.tile(np.nan, len(shipDF))

    # # Calculate delta pco2 based on chosen float parameter
    # floatDF['pco2'] = floatDF.reset_index()[('pCO2_' + float_param)]
    # floatDF['delta_pco2'] = floatDF['pco2'] - floatDF['pco2_atm']
    # floatDF

    # Output of 2.0. Has mld, adt, pco2 
    filepath = '../working-vars/regression/inputs/test-bgcArgo/'
    test_bgcArgo_2024 = pd.read_csv(filepath + 'test_bgcArgo2024_processed_pCO2_pHbias5_pK1_mld_adt_acc20260211.csv', index_col=0) 

    return [floatDF, shipDF]
    # = pd.read_csv('../working-vars/argo/processed-SOCCOM-20m/soccomDF_avg20m_withMLD_ADT_acc20260203.csv', index_col=0)


def import_preprocessed_inputs(pcm_params='pc8_gmm6', float_param='pHbias5_pK1'):
    """ 
     Still using! 
     Output of 2.0_RUN
    Data already clustered, regression targets and features added"""

    filepath = '../working-vars/regression/inputs/' + pcm_params + '/'

    if pcm_params == 'pc8_gmm8':
        datetag = '20260123'
        # floatname = 'floatDF_ADT_SLA_soccom20m_pCO2_'+ float_param + '_PCM_' + pcm_params + '_acc' + datetag + '.csv'
        # shipname = 'shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_PCM_' + pcm_params + '_acc' + datetag + '.csv'
    elif pcm_params == 'pc8_gmm7':
        datetag = '20260204'
        # floatname = 'floatDF_ADT_SLA_soccom20m_pCO2_pHbias5_pK1_PCM_' + pcm_params + '_acc' + datetag + '.csv'
        # shipname = 'shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_PCM_' + pcm_params + '_acc' + datetag + '.csv'
    elif pcm_params == 'pc8_gmm6':
        datetag = '20260123'

    floatname = 'floatDF_ADT_SLA_soccom20m_pCO2_'+ float_param + '_PCM_' + pcm_params + '_acc' + datetag + '.csv'
    shipname = 'shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_PCM_' + pcm_params + '_acc' + datetag + '.csv'
    
    
    floatDF = pd.read_csv(filepath + floatname, index_col=0)
    shipDF = pd.read_csv(filepath + shipname, index_col=0)

    return [floatDF, shipDF]

# def import_regression_inputs(addSatelliteFeatures = False, pcm_params = 'pc8_gmm8'):
#     """ 
#     Files were created in 2.0_process_regression_data.ipynb
#     These have all variables for regression, including target pco2
#     Used as inputs in notebooks 2.1 – 
    
#     SOCAT fco2 has been converted to pco2 using NCEP atmospheric pressures
#     """

#     # These have pco2 and delta-pco2 (ocean minus atmosphere, in uatm)
#     filepath = '../working-vars/regression/inputs/'
#     float_parameter = 'pHbias5_pK1'
#     datetag = '20260119' #20251209'
#     soccomDF = pd.read_csv(filepath + 'floatDF_soccom20m_pco2_' + float_parameter + '_PCM_8pc8class_acc' + datetag + '.csv',
#     index_col=0)

#     datetag = '20260111' #'20260111'
#     socatDF = pd.read_csv(filepath + 'shipDF_socatv2024_1d_pCO2converted_co2sys_PCM_8pc8class_acc' + datetag + '.csv',
#     index_col=0)

#     if addSatelliteFeatures:
#         # return output of 2.1_add_regression_features.ipynb
#         print('Using data with added satellite features (ADT, SLA)')
#         # datetag = '20260112'
#         # soccomDF = pd.read_csv(filepath + 'floatDF_ADT_SLA_soccom20m_pco2_pHbias3_pK1_PCM_8pc8class_acc' + datetag + '.csv', 
#         #                     index_col=0)
#         datetag = '20260119'
#         soccomDF = pd.read_csv(filepath + 'floatDF_ADT_SLA_soccom20m_pco2_pHbias5_pK1_PCM_8pc8class_acc' + datetag + '.csv', 
#                             index_col=0)
#         datetag = '20260112'
#         socatDF = pd.read_csv(filepath + 'shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_PCM_8pc8class_acc' + datetag + '.csv', 
#                               index_col=0)

#     return [soccomDF, socatDF]

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

def import_soccom_20m_averages(processed=True, start_date = '2013-12-31', end_date = '2023-12-31'):
    """ 
    Data to be used in 2.0 and 3.1. Courtesy Alison Gray
    Surface data from SOCCOM averaged over uppermost 20m
    pHbias3_pK1 represents pH adjusted by 3 mpH and pK1 adjusted
    """
    if processed == False:
        directory_path = '/Volumes/crusoe-repo/data/SOCCOM-GOBGC-20m-averages-v2/'
        contents = os.listdir(directory_path)

        # temp = sio.loadmat(var_pK1_20_1902303.mat')
        # use_cols = [x[0][0] for x in temp['var_names']] + ['wmoid']

        soccom_df = pd.DataFrame() #columns=use_cols)
        for x in contents:
            if x.startswith('var_pK1_20_'):
                temp = sio.loadmat(directory_path + x)
                profnum = x[-11:-4]

                wmo_df = pd.DataFrame(columns=[x[0][0] for x in temp['var_names']], data = temp['var'])
                wmo_df['wmoid'] = np.tile(profnum, len(wmo_df))

                wmo_df['id'] = wmo_df.index.values
                # 'obid' is temporary because it may not be the same as profid in bgcINDEX if rows were dropped
                wmo_df['obid'] = wmo_df.apply(lambda row: str(row.wmoid) + '_id' + str(row.id), axis=1)

                soccom_df = pd.concat([soccom_df, wmo_df], ignore_index=True)

        def datenum_to_datetime(matlab_datenum):
            days = matlab_datenum % 1
            dt = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=days) - timedelta(days=366)
            return dt

        soccom_df['datetime'] = soccom_df['datenum'].apply(lambda x: datenum_to_datetime(x))
        soccom_df = soccom_df[soccom_df.Lat<-35]
        # NEW RESTRICT TIME
        soccom_df = soccom_df[(soccom_df.datetime>np.datetime64(start_date)) & (soccom_df.datetime<np.datetime64(end_date))]

        # process 20m averages
        soccom_df['prof_datetag'] = soccom_df.datetime.astype(str)
        soccom_df['prof_datetag'] = soccom_df['prof_datetag'].apply(lambda x: x.replace('-','').split(' ')[0])
        soccom_df['prof_datetag'] = soccom_df.apply(lambda row: str(row['wmoid']) + '_' + str(row['prof_datetag']), axis=1)
        soccom_df = soccom_df.reset_index()

    if processed == True:
        soccom_df = pd.read_csv('../working-vars/argo/processed-SOCCOM-20m/soccomDF_avg20m_withMLD_ADT_acc20260203.csv', index_col=0)

    return soccom_df



def main():
    print('Loading data...')

if __name__ == "__main__":
    main()
# %%
