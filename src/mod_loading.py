import xarray as xr
import pandas as pd
import numpy as np
import mod_ocean
import gsw
import datetime
from datetime import datetime, timedelta
import os
import scipy.io as sio
# from mod_ocean import datetime2ytd
import mod_preprocessing as mod_prep

# [coreDS, coreINDEX] = loader.import_core_data(type = 'L3_only')
# [bgcDS, bgcINDEX] = loader.import_bgc_data(type = 'L3_only')
def print_update():
    print('loaded now')

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

    elif type == 'processed': # # with MLD added
        # coreINDEX, coreDS = mod_prep.add_mixedlayer_pressure(coreINDEX, coreDS)
        # datetag = '20260121'
        # folder = '../working-vars/argo/import-L3/'
        # coreDS = xr.open_dataset(folder + 'coreDATA_validL3_2014-2023_withMLD_acc' + datetag + '.nc')

        # To use coreDS with satellite ADT and MLD added (from 2.0_Preprocessing)
        # coreINDEX processed has all regression vars, including atmospheric pco2 (converted from ppm)
        datetag = '20260201'
        folder = '../working-vars/regression/inputs/core/'
        coreDS = xr.open_dataset(folder + 'coreDATA_processed_2014-2023_MLD_ADT_SLA_acc' + datetag + '.nc')
        coreINDEX = xr.open_dataset(folder + 'coreINDEX_pco2-atmos_processed_2014-2023_MLD_ADT_acc20260211.nc')

        # coreINDEX['mld'] = coreDS['mld'].mean(dim='pressure')


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
        datetag = '20260121'
        bgcINDEX = xr.open_dataset(folder + 'bgcINDEX_validL3_2014-2023_withMLD_acc' + datetag + '.nc')
        bgcDS = xr.open_dataset(folder + 'bgcDATA_validL3_2014-2023_withMLD_acc' + datetag + '.nc')
    
    elif type == 'test':
        folder = '/Volumes/crusoe-repo/data/bgc/L3-interp/'
        datetag = '20260209'
        bgcINDEX = xr.open_dataset(folder + 'bgcINDEX_valid_interp_2024_TESTING_acc' + datetag + '.nc')
        bgcDS = xr.open_dataset(folder + 'bgcDATA_valid_interp_2024_TESTING_acc' + datetag + '.nc')

    return [bgcDS, bgcINDEX]



# def import_float_20m_data():


def import_socat_L2():
    # Masked to open ocean south of 35S
    filepath = '/Volumes/crusoe-repo/data/socat/L2-mask/' 

    # new resample at 1d resolution jan 2 in 0.1_socat2024 processing
    socat = xr.open_dataset(filepath + 'SOCATv2024_SO_1d_open_ocean_INDEX_acc20260102.nc')

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
    Used for classification in 1.2_pcm_classify_bgcArgo_ship.ipynb
    Files were created in 0.5_socat2024_colocation.ipynb
    """
    filepath = '../working-vars/socat/colocate-coreArgo/'
    sepdict_7d = {key:None for key in [str(x) for x in range(2014,2024)]}

    for x in os.listdir(filepath):
        if x.startswith('colocate_7d') & x.endswith('20260111.csv'):
            sepdict_7d[x[14:18]] = pd.read_csv(filepath+x, index_col=0)
            # print('Imported data for _7d window: ' + x)

    sepstat_7d = pd.concat(sepdict_7d.values()).reset_index().drop(['level_0', 'index'], axis=1)

    return sepstat_7d

def import_clustering_results(pcm_params='pc8_gmm8'):
    """ 
    Files were created in 1.1_pcm_fit_coreArgo.ipynb
    [Y_gmm, allprobs, gmm_desc, coreDF_ave] = loader.import_clustering_results()
    ~ 2 sec
    """
    filepath = '../working-vars/pcm/' + pcm_params + '/'

    if pcm_params == 'pc8_gmm8':
        # from dec 2025
        Y_gmm = pd.read_csv(filepath + 'Y_gmm_20251201.csv', index_col=0) # Results of GMM
        datetag = '20251208'
        allprobs = pd.read_csv(filepath + 'postprobs_allclasses_' + datetag + '.csv')
        allprobs = allprobs.rename(columns = {str(k):(k+1) for k in range(8)})
    elif pcm_params == 'pc8_gmm6':
        # updated jan 20226
        datetag = '20260119'
        Y_gmm = pd.read_csv(filepath + 'Y_gmm_501dbar_' + pcm_params + '_' + datetag + '.csv',
                             index_col=0) # Results of GMM
        allprobs = pd.read_csv(filepath + 'postprobs_allclasses_' + datetag + '.csv')
        allprobs = allprobs.rename(columns = {str(k):(k+1) for k in range(8)})

    elif pcm_params == 'pc8_gmm7':
        datetag = '20260201'
        Y_gmm = pd.read_csv(filepath + 'Y_gmm_501dbar_' + pcm_params + '_' + datetag + '.csv',
                             index_col=0) # Results of GMM
        allprobs = pd.read_csv(filepath + 'postprobs_allclasses_' + datetag + '.csv')  # already renamed
        
    coreINDEX = import_core_data(type='L3_only')[1]

    coreDF_ave = coreINDEX.to_dataframe()
    coreDF_ave['prof_datetag'] = coreDF_ave.datetime.astype(str)
    coreDF_ave['prof_datetag'] = coreDF_ave['prof_datetag'].apply(lambda x: x.replace('-','').split(' ')[0])
    coreDF_ave['prof_datetag'] = coreDF_ave.apply(lambda row: str(int(row['wmoid'])) + '_' + str(row['prof_datetag']), axis=1)

    return [Y_gmm, allprobs, pcm_params, coreDF_ave]

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
    with MLD and ADT added. updated feb 3 2026
    Files created in 2.0_RUN, originally after clustering
    Saving output into new foloder processed-< > so MLD 
    doesn't get recalculated every time with new clustering
    """
    floatDF = import_soccom_20m_averages(processed=True)
    shipDF = pd.read_csv('../working-vars/socat/processed-socat2024/shipDF_ADT_SLA_socatv2024_1d_pCO2converted_co2sys_acc20260203.csv', index_col=0)
    shipDF['cluster'] = np.tile(np.nan, len(shipDF))

    # Calculate delta pco2 based on chosen float parameter
    floatDF['pco2'] = floatDF.reset_index()[('pCO2_' + float_param)]
    floatDF['delta_pco2'] = floatDF['pco2'] - floatDF['pco2_atm']
    # floatDF 
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