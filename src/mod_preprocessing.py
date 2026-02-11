import pandas as pd
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
import os
from datetime import datetime, timedelta
import mod_loading as loader

import mod_argo
import mod_ocean
import gsw

# def print_dict_counts(plat_dict):

# %% 2.0_process_pco2_data

def add_mixedlayer_pressure(platINDEX, platDS):
    """ Originally from 2.0_process_pco2_data
    
    
    """
    # # Calculate mixed layer pressure 
    plat_mlps, no_data = mod_argo.calc_mlp(platDS, threshold=0.03)
    plat_mlps['mld'] = gsw.z_from_p(plat_mlps.mlp.values, platINDEX.sel(profid=plat_mlps.index).latitude.values)

    # Select out only valid
    platINDEX = platINDEX.sel(profid=plat_mlps.index)
    platDS = platDS.sel(profid=plat_mlps.index)

    # Add mixed layer depth (mld)
    platINDEX['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
    platDS['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative

    # Add mixed layer pressure (mlp)
    platINDEX['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')
    platDS['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')

    return platINDEX, platDS

# %% ADD TARGET VARIABLE FOR ML INPUTS (delta-pco2)
# Within 2.0 Preprocessing we run:
    # shipDF = mod_prep.add_regression_time_vars(shipDF)
    # shipDF = mod_prep.add_regression_carbon_vars(shipDF, convert_pco2=True) 
    # shipDF['delta_pco2'] = shipDF['pco2_ocean'] - shipDF['pco2_atmos']


def add_regression_time_vars(platDF):
    # Add new time variables
    platDF['linear_time'] = platDF.datetime.apply(lambda x: mod_ocean.datetime2ytd(np.datetime64(x), ref_time='2014-01-01'))
    platDF['ydcos']= mod_ocean.get_ydsines(platDF.linear_time.values)[0]
    platDF['ydsin']= mod_ocean.get_ydsines(platDF.linear_time.values)[1]
    platDF = mod_ocean.add_decimalyr(platDF)

    return platDF

def add_regression_carbon_vars(platDF, convert_pco2=True): # , make_delta = True):
    """ Add marine boundary layer xCO2 (ppm) to dataset
    """
    if 'atmos_pres_atm' not in platDF.columns:
        platDF['atmos_pres_Pa'] = platDF.apply(lambda row: nearest_slp(row), axis=1)
        platDF['atmos_pres_atm'] = platDF['atmos_pres_Pa'] / 1013.25 / 100

    if 'decimalyr' not in platDF.columns:
        platDF = mod_ocean.add_decimalyr(platDF)

    # Get atmospheric fco2 from MBL marine boundary layer product
    # make target delta pco2 variable
    platDF['sin_lat'] = np.sin(platDF['latitude']*np.pi/180)
    platDF['atmos_co2_ppm'] = platDF.apply(get_row_atmos_co2_ppm, axis=1)
    # if make_delta:
    #     platDF['delta_pco2'] = platDF['pco2'] - platDF['pco2_atm']
    
    if convert_pco2:
        platDF = convert_co2_ppm_to_pco2(platDF)

    return platDF


# Add target variable
def get_row_atmos_co2_ppm(row):
    mbl_co2 = xr.open_dataset('../data/marineboundarylayer/co2_mbl_2014-2023_dataset_combined.nc')
    mbl_co2.lat_deg.values

    # originally in ppm
    result_ppm = mbl_co2.sel(decimalyr=row.decimalyr, method='nearest').interp(sin_lat=row.sin_lat)['mean'].values

    return result_ppm

def get_row_atmos_vapor_pres(row):
    """ used in marine boundary layer xCO2 (ppm) to pco2 (uatm) conversion 
    Zeebe and Wolf-Gladrow 2001
    """
    sst_kelvin = row.sst + 273.15
    ln_vapor_pres = (24.4542 
               - 67.4509 * (100 / sst_kelvin) 
               - 4.8489 * np.log(sst_kelvin / 100) 
               - 0.000544 * row.sss)
    vapor_pres = np.exp(ln_vapor_pres)
    return vapor_pres

def nearest_slp(row):
        # Add atmospheric pressure from NCEP sea level pressure data
        # Needed for conversion from fco2 to pco2
        # Returns values in Pascals (divide by 101325 to get to atm)
        year = row.datetime.year
        filepath = '/Volumes/crusoe-repo/data/ncep_slp/'
        ncep_sea_level_pressure = xr.open_dataset(filepath + 'slp.' + str(year) + '.nc')
        return ncep_sea_level_pressure.sel(lon=row.lon_round, lat=row.lat_round, time = row.datetime,
                            method='nearest')['slp'].values.tolist()


def convert_co2_ppm_to_pco2(platDF, ppm_col='atmos_co2_ppm'):
    """ convert ppm to pco2 using Zeebe and Wolf-Gladrow 2001 formulation"""
    platDF['sst'] = gsw.t_from_CT(platDF['SA'], platDF['CT'], (np.tile(0, len(platDF))))
    platDF['sss'] = gsw.SA_from_SP(platDF['SA'], (np.tile(0, len(platDF))), platDF['longitude'], platDF['latitude'])

    platDF['vapor_pres_atm'] = platDF.apply(lambda row: get_row_atmos_vapor_pres(row), axis=1)
    # platDF['pco2_atmos'] = platDF.apply(lambda row: get_atmos_pco2(row), axis=1) # in uatm 
    platDF['pco2_atmos'] = platDF.apply(lambda row: row.atmos_co2_ppm * (row.atmos_pres_atm - row.vapor_pres_atm), 
                                        axis=1) # in uatm 

    return platDF

# %% SOCAT FUNCTIONS (2.0)


def convert_socat_fco2(socatDS):
    """ 
    Convert socat fCO2 to pCO2 using atmospheric pressure from NCEP sea level pressure data
    socatDS: xarray Dataset of SOCAT with 'expoID'
    """
    socatDF = socatDS.to_dataframe()
    socatDF['datetime'] = socatDF.datetime.astype('datetime64[ns]')

    # add 
    socatDF['lon_round'] = round((socatDF['longitude'] + 180)/2.5)*2.5
    socatDF['lat_round'] = round(socatDF['latitude']/2.5)*2.5
    socatDF['cruiseID'] = socatDF['expoID'].apply(lambda x: x.split('_')[0])

    # ncep_sea_level_pressure = xr.open_dataset('/Users/sangminsong/Downloads/slp.2023.nc')
    # Using https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html


    # ncep_sea_level_pressure = xr.open_dataset('/Users/sangminsong/Downloads/slp.2023.nc')
    socatDF['atmos_pres_Pa'] = socatDF.apply(lambda row: nearest_slp(row), axis=1)
    socatDF['atmos_pres_atm'] = socatDF['atmos_pres_Pa'] / 1013.25 / 100

    # Convert fco2 to pco2
    socatDF['pco2_ocean'] = socatDF.apply(lambda row: pyco2.sys(
                                par1 = row['fco2rec'], 
                                par1_type = 5,
                                temperature = row.sst,  
                                pressure=0,
                                temperature_out = row.sst, # need to specify either temp or pres_out
                                pressure_atmosphere_out= row.atmos_pres_atm # from ncep
                                )['pCO2'], axis=1)
    socatDF = socatDF.rename(columns={'yearday': 'linear_time'})

    return socatDF

# %% SOCCOM 20m AVE PROCESSING FUNCTIONS

def process_soccom_20m_clusters(bgcDS, use_var='pCO2_pHbias5_pK1', start_date = '2013-12-31', end_date = '2023-12-31'): #, use_cols=None):
    """ 
    Select which variable (carbon correction) to use from the 20m averages
    Fills in missing clusters
    Uses NCEP sea level pressure to assign atmospheric pressure (in Pa and in uatm)
    
    """
    soccom_df = loader.import_soccom_20m_averages(processed=False, start_date=start_date, end_date=end_date)

    # from core clustered data
    bgcClasses = bgcDS.to_dataframe().reset_index()
    bgcClasses['prof_datetag'] = bgcClasses['profid'].apply(lambda x: str(x).split('_')[0])
    bgcClasses['prof_datetag'] = bgcClasses.apply(lambda row: row.prof_datetag + '_' + row.datetime.strftime('%Y%m%d'), axis=1)
    bgcClasses_index = bgcClasses.reset_index().groupby('prof_datetag').first()
    bgcClasses_index

    # Assign clusters by comparison to bgcINDEX classes
    soccom_df['cluster'] = soccom_df['prof_datetag'].apply(lambda x: bgcClasses_index.loc[str(x)]['cluster'] if x in bgcClasses_index.index.tolist() else np.nan)

    # Assign profid that corresponds to the bgcINDEX using the WMO number and date (prof_datetag)
    soccom_df['profid'] = soccom_df['prof_datetag'].apply(lambda x: bgcClasses_index.loc[str(x)]['profid'] if x in bgcClasses_index.index.tolist() else np.nan)

    print('Clusters assigned.')

    # Fill in missing  cluster values where we can -- 
    # look for same float, profiles within +/- 11 days
    # If cluster was same before and after, assign to that cluster
    soccom_df_filled = soccom_df.set_index('prof_datetag')
    missing = soccom_df[soccom_df.cluster.isna()]
    clustered_datetags = bgcClasses_index.reset_index().prof_datetag.unique()
    print('Initially missing clusters for ' + str(len(missing)) + ' profiles, filling in ==>')

    profsReplaced = []
    # for ind in range(len(missing)):
    for ind, tag in enumerate(missing.prof_datetag.unique()):
        obs = missing.iloc[ind, :]
        inds = [x for x in clustered_datetags if x.startswith(obs.wmoid)]
        findDF = soccom_df.reset_index()
        same_float = findDF[findDF.prof_datetag.isin(inds)].copy()

        if len(same_float)>0:
            same_float.loc[:,'timediff'] = same_float.apply(lambda row: abs((row.datetime - obs.datetime).days), axis=1)
            valid = same_float[same_float.timediff<12]
            if valid.cluster.nunique() == 1:
                soccom_df_filled.loc[tag, 'cluster'] = valid.cluster.values[0]
                profsReplaced = profsReplaced + [tag]

    print('Filled in ' + str(len(profsReplaced)) + ' missing clusters using adjacent profiles')
    soccom_df_filled = soccom_df_filled.rename(columns={'Lat':'latitude', 'Lon':'longitude',
                        'MLD':'mld', 'Absolute_Salinity':'SA', 'ConservTemp':'CT'})

    # Add sea level pressure 
    soccom_df_filled['lon_round'] = round((soccom_df_filled['longitude'] + 180)/2.5)*2.5
    soccom_df_filled['lat_round'] = round(soccom_df_filled['latitude']/2.5)*2.5
    soccom_df_filled['atmos_pres_Pa'] = soccom_df_filled.apply(lambda row: nearest_slp(row), axis=1)
    soccom_df_filled['atmos_pres_atm'] = soccom_df_filled['atmos_pres_Pa'] / 1013.25 / 100

    return soccom_df_filled

def datenum_to_datetime(matlab_datenum):
  days = matlab_datenum % 1
  dt = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=days) - timedelta(days=366)
  return dt




    

# # %% MOVED ALL TO MOD_REGRESSION

# def print_obs_counts(training_classes, validation_classes):
#     """  
#     originally from 3.1_clustered_rfr.ipynb

#     @param  training_classes: dict of training dataframes for each cluster

#             validation_classes: dict of validation dataframes for each cluster"""
#     countobs = pd.DataFrame({'cluster':range(1,9), 
#                         'train':[len(k) for k in training_classes.values()],
#                         'validation':[len(k) for k in validation_classes.values()]},
#                         # 'train_float':[len(k[~k['wmoid'].isna()]) for k in training_classes.values()],
#                         # 'train_ship':[len(k[k.wmoid.isna()]) for k in training_classes.values()],
#                         # 'val_float':[len(k[~k.wmoid.isna()]) for k in validation_classes.values()],
#                         # 'val_ship':[len(k[k.wmoid.isna()]) for k in validation_classes.values()]
#                          )


#     countobs['train_float'] = [len(k[~k['wmoid'].isna()]) for k in training_classes.values()]
#     countobs['train_ship'] = [len(k[k.wmoid.isna()]) for k in training_classes.values()]

#     countobs['val_float'] = [len(k[~k['wmoid'].isna()]) for k in validation_classes.values()]
#     countobs['val_ship'] = [len(k[k.wmoid.isna()]) for k in validation_classes.values()]

#     # countobs['train_pc_float'] = countobs.apply(lambda row: row['train_float']/row['train']*100, axis=1)
#     countobs['train_pc_float'] = countobs.apply(lambda row: row['train_float']/row['train']*100, axis=1)
#     countobs['val_pc_float'] = countobs.apply(lambda row: row['val_float']/row['validation']*100, axis=1)

#     # print('percent split into training/validation: ' + str(len(trainDF_all)/(len(trainDF_all)+len(valDF_all))*100))
#     countobs.set_index('cluster', inplace=True)

#     return countobs


# def make_feat_dict(feat_lists):
#     """ 
#     @param    feat_lists: list of lists of feature names for each cluster
#     """
#     ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     labels = ['feat'+ ascii_uppercase[i] for i in range(len(feat_lists))]
#     feat_options = {labels[i]: feat_lists[i] for i in range(len(feat_lists))}

#     return feat_options 

# def make_run_tags(feat_options, data_options, target_options):
#     """ 
#     @ param   feat_options: dict of feature lists (make_feat_dict())
#                data_options: dict of datasets for training, validation
#                     data_options = {'float': [trainClasses_float, valClasses],
#                         'ship': [trainClasses_ship, valClasses],
#                         'combined': [trainClasses, valClasses]}


#                target_options: list of target variable names"""
#     # Automatically generate run tag combinations
#     run_tags = []
#     for key1 in feat_options.keys():
#         for key2 in data_options.keys():
#             for target in target_options:
#                 run_tags.append(key1 + '-' + key2 + '-' + target)
#     return run_tags
