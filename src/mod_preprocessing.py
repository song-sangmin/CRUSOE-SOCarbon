import pandas as pd
import xarray as xr
import numpy as np
import PyCO2SYS as pyco2
import pyproj
import os
from datetime import datetime, timedelta
import mod_loading as loader

import mod_argo
import mod_ocean
import gsw
from tqdm import tqdm

# def print_dict_counts(plat_dict):
# %% miscellaneous useful

def print_bounds(platDF):
    # print('Bounds of data: \n')
    print('Sample count: ' + '\t' + str(len(platDF)))
    print('\tDates: \t' + str(platDF.datetime.min()) + ' to ' + str(platDF.datetime.max()))
    # print('Latitude:\t' + str(platDF.latitude.min()) + ' to ' + str(platDF.latitude.max()))
    # print('Longitude:\t' + str(platDF.longitude.min()) + ' to ' + str(platDF.longitude.max()))


def match_profid_from_datetag(soccomDF, floatINDEX):
    """ 
    used to be assign_profid_from_datetag
    Match SOCCOM 20m averages with profid from bgcINDEX or coreINDEX 
    since these are from another source 
    @param      soccomDF with a prof_datetag column
                floatINDEX has coordinated 'profid'
    
    """
    # Add prof_datetag to the dataset for matching
    finder_index = index_by_prof_datetag(floatINDEX.to_dataframe())

    # soccomDF = soccomDF.reset_index() #set_index('prof_datetag')
    soccomDF['profid'] = soccomDF['prof_datetag'].apply(lambda x: finder_index.loc[str(x)]['profid'] if x in finder_index.index.tolist() else np.nan)
    soccomDF = soccomDF.dropna(subset=['profid']).copy()
    soccomDF.set_index('profid', inplace=True)

    return soccomDF

def add_datetag_from_profid(platDF, floatINDEX):
    datetag_finder = index_by_prof_datetag(floatINDEX.to_dataframe()).reset_index().set_index('profid')
    platDF['prof_datetag'] = platDF['profid'].apply(lambda x: datetag_finder.loc[str(x)]['prof_datetag'] if str(x) in datetag_finder.index.tolist() else np.nan)


def index_by_prof_datetag(platDF):
    """ 
    bgcINDEX is an xr dataset
    make pd Dataframe indexed by prof_datetag so you can assign consistent profids to soccom 20m aves"""
    # finder = bgcINDEX.to_dataframe().reset_index()
    finder = platDF.reset_index()
    finder['datetime'] = finder['datetime'].astype('datetime64[ns]')
    finder['prof_datetag'] = finder['profid'].apply(lambda x: str(x).split('_')[0])
    finder['prof_datetag'] = finder.apply(lambda row: row.prof_datetag + '_' + row.datetime.strftime('%Y%m%d'), axis=1)
    # finder['prof_datetag'] = finder.apply(lambda row: row.prof_datetag + '_' + row.datetime[:10].replace('-',''), axis=1)
    # finder_index = finder.reset_index().groupby('prof_datetag').first()

    return finder

def add_mixedlayer_pressure(platINDEX, platDS):
    """ Originally from 2.0_process_pco2_data
    """
    # Calculate mixed layer pressure 
    plat_mlps, no_data = mod_argo.calc_mlp(platDS, threshold=0.03)
    plat_mlps['mld'] = gsw.z_from_p(plat_mlps.mlp.values, platINDEX.sel(profid=plat_mlps.index).latitude.values)

    # Select out only valid
    platINDEX = platINDEX.sel(profid=plat_mlps.index)
    platDS = platDS.sel(profid=plat_mlps.index)

    platINDEX['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
    platDS['mld']= -xr.DataArray(plat_mlps.mld.values, dims='profid') # Notice negative
    platINDEX['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')
    platDS['mlp'] = xr.DataArray(plat_mlps.mlp.values, dims='profid')

    return platINDEX, platDS


def lon_180_to_360(platDF):
    """ Convert longitude from -180 to 180 to 0 to 360 for matching with NCEP sea level pressure data
    """
    platDF['longitude'] = platDF['longitude'].apply(lambda x: x if x >=0 else x + 360)
    return platDF

# %% P1-processed : Making pd Dataframes ready for regression

# ADD TARGET VARIABLE FOR ML INPUTS (delta-pco2)
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
    if 'lon_round' not in platDF.columns:
            platDF['lon_round'] = round((platDF['longitude'] + 180)/2.5)*2.5
            platDF['lat_round'] = round(platDF['latitude']/2.5)*2.5

    if 'atmos_pres_atm' not in platDF.columns:
        platDF['atmos_pres_Pa'] = platDF.apply(lambda row: get_row_sealevelpressure(row), axis=1)
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

# %% ATMOSPHERIC AND OCEANIC CARBON VARS
# Add target variable
def get_row_atmos_co2_ppm(row):
    mbl_co2 = xr.open_dataset('../data/marineboundarylayer/co2_mbl_2014-2023_dataset_combined.nc')
    mbl_co2.lat_deg.values

    # originally in ppm
    result_ppm = mbl_co2.sel(decimalyr=row.decimalyr, method='nearest').interp(sin_lat=row.sin_lat)['mean'].values

    return result_ppm

def get_row_atmos_vapor_pres(row):
    """ used in marine boundary layer xCO2 (ppm) to pco2 (uatm) conversion 
    Requires data to have sst and sss 
    Zeebe and Wolf-Gladrow 2001
    """
    sst_kelvin = row.sst + 273.15
    ln_vapor_pres = (24.4542 
               - 67.4509 * (100 / sst_kelvin) 
               - 4.8489 * np.log(sst_kelvin / 100) 
               - 0.000544 * row.sss)
    vapor_pres = np.exp(ln_vapor_pres)
    return vapor_pres

def get_row_sealevelpressure(row):
    # used to be nearest_slp(row)
    # Add atmospheric pressure from NCEP sea level pressure data
    # Needed for conversion from fco2 to pco2
    # Returns values in Pascals (divide by 101325 to get to atm)
    year = row.datetime.year
    filepath = '/Volumes/crusoe-repo/data/ncep_slp/'
    ncep_sea_level_pressure = xr.open_dataset(filepath + 'slp.' + str(year) + '.nc')

    atmos_pres_Pa = ncep_sea_level_pressure.sel(lon=row.lon_round, lat=row.lat_round, time = row.datetime,
                        method='nearest')['slp'].values.tolist()
    return atmos_pres_Pa


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


def convert_socat_fco2(socatDF):
    """ 
    Convert socat fCO2 to pCO2 using atmospheric pressure from NCEP sea level pressure data
    # socatDS: xarray Dataset of SOCAT with 'expoID'
    socatDF: in 2.0 preprocessing use colocation DF from 0.5_
    """
    # socatDF = socatDS.to_dataframe()
    socatDF['datetime'] = socatDF.datetime.astype('datetime64[ns]')

    # Calculate SA, CT for vapor pressure
    socatDF['SA'] = gsw.SA_from_SP(socatDF['sal'],  (np.tile(0, len(socatDF))), socatDF['longitude'], socatDF['latitude'])
    socatDF['CT'] = gsw.CT_from_t(socatDF['SA'], socatDF['sst'], (np.tile(0, len(socatDF))))

    # add 
    socatDF['lon_round'] = round((socatDF['longitude'] + 180)/2.5)*2.5
    socatDF['lat_round'] = round(socatDF['latitude']/2.5)*2.5
    socatDF['cruiseID'] = socatDF['expoID'].apply(lambda x: x.split('_')[0])

    # ncep_sea_level_pressure = xr.open_dataset('/Users/sangminsong/Downloads/slp.2023.nc')
    # Using https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html


    # ncep_sea_level_pressure = xr.open_dataset('/Users/sangminsong/Downloads/slp.2023.nc')
    socatDF['atmos_pres_Pa'] = socatDF.apply(lambda row: get_row_sealevelpressure(row), axis=1)
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

def process_soccom_20m_averages(start_date = '2013-12-31', end_date = '2023-12-31', use_var='pCO2_pHbias5_pK1'): #, use_cols=None):
    """ 
    Select which variable (carbon correction) to use from the 20m averages
    Fills in missing clusters
    Uses NCEP sea level pressure to assign atmospheric pressure (in Pa and in uatm)
    
    """
    soccom_df = loader.import_soccom_20m_averages(processed=False, start_date=start_date, end_date=end_date)
    soccom_df = soccom_df.rename(columns={'Lat':'latitude', 'Lon':'longitude',
                        'MLD':'mld', 'Absolute_Salinity':'SA', 'ConservTemp':'CT'})
    
    if 'lon_round' not in soccom_df.columns: # for slp
        soccom_df['lon_round'] = round((soccom_df['longitude'] + 180)/2.5)*2.5
        soccom_df['lat_round'] = round(soccom_df['latitude']/2.5)*2.5
    soccom_df['atmos_pres_Pa'] = soccom_df.apply(lambda row: get_row_sealevelpressure(row), axis=1)
    soccom_df['atmos_pres_atm'] = soccom_df['atmos_pres_Pa'] / 1013.25 / 100

    print('Imported SOCCOM pCO2 data using variable ' + use_var)
    soccom_df['pco2_ocean'] = soccom_df[use_var]

    # soccom_df['prof_datetag'] = soccom_df['profid'].apply(lambda x: str(x).split('_')[0])
    # bgcClasses['prof_datetag'] = bgcClasses.apply(lambda row: row.prof_datetag + '_' + row.datetime.strftime('%Y%m%d'), axis=1)
    # bgcClasses_index = bgcClasses.reset_index().groupby('prof_datetag').first()

    # bgcClasses_index


    return soccom_df


# def assign_soccom_20m_clusters(soccom_df, clusteredDS):
#     """  used to be within the process_soccom_20m_clusters function 
#      use with clusteredDS after running PCM 
#      clusteredDS should already have mld, adt, atmospheric pco2 added to it. 
#     """
#     # from core clustered data
#     # bgcClasses = clusteredDS.to_dataframe().reset_index()
#     # bgcClasses['prof_datetag'] = bgcClasses['profid'].apply(lambda x: str(x).split('_')[0])
#     # bgcClasses['prof_datetag'] = bgcClasses.apply(lambda row: row.prof_datetag + '_' + row.datetime.strftime('%Y%m%d'), axis=1)
#     # bgcClasses_index = bgcClasses.reset_index().groupby('prof_datetag').first()
#     bgcClasses_index = get_index_prof_datetag(clusteredDS.to_dataframe().reset_index())

#     # Assign clusters by comparison to bgcINDEX classes
#     soccom_df['cluster'] = soccom_df['prof_datetag'].apply(lambda x: bgcClasses_index.loc[str(x)]['cluster'] if x in bgcClasses_index.index.tolist() else np.nan)

#     # Assign profid that corresponds to the bgcINDEX using the WMO number and date (prof_datetag)
#     soccom_df['profid'] = soccom_df['prof_datetag'].apply(lambda x: bgcClasses_index.loc[str(x)]['profid'] if x in bgcClasses_index.index.tolist() else np.nan)

#     print('Clusters assigned.')

#     # Fill in missing  cluster values where we can -- 
#     # look for same float, profiles within +/- 11 days
#     # If cluster was same before and after, assign to that cluster
#     soccom_df_filled = soccom_df.set_index('prof_datetag')
#     missing = soccom_df[soccom_df.cluster.isna()]
#     clustered_datetags = bgcClasses_index.reset_index().prof_datetag.unique()
#     print('Initially missing clusters for ' + str(len(missing)) + ' profiles, filling in ==>')

#     profsReplaced = []
#     # for ind in range(len(missing)):
#     for ind, tag in enumerate(missing.prof_datetag.unique()):
#         obs = missing.iloc[ind, :]
#         inds = [x for x in clustered_datetags if x.startswith(obs.wmoid)]
#         findDF = soccom_df.reset_index()
#         same_float = findDF[findDF.prof_datetag.isin(inds)].copy()

#         if len(same_float)>0:
#             same_float.loc[:,'timediff'] = same_float.apply(lambda row: abs((row.datetime - obs.datetime).days), axis=1)
#             valid = same_float[same_float.timediff<12]
#             if valid.cluster.nunique() == 1:
#                 soccom_df_filled.loc[tag, 'cluster'] = valid.cluster.values[0]
#                 profsReplaced = profsReplaced + [tag]

#     print('Filled in ' + str(len(profsReplaced)) + ' missing clusters using adjacent profiles')
    

#     # # Add sea level pressure 
#     # soccom_df_filled['lon_round'] = round((soccom_df_filled['longitude'] + 180)/2.5)*2.5
#     # soccom_df_filled['lat_round'] = round(soccom_df_filled['latitude']/2.5)*2.5
#     # soccom_df_filled['atmos_pres_Pa'] = soccom_df_filled.apply(lambda row: get_row_sealevelpressure(row), axis=1)
#     # soccom_df_filled['atmos_pres_atm'] = soccom_df_filled['atmos_pres_Pa'] / 1013.25 / 100

#     return soccom_df_filled


# %% SATELLITE ADT

def add_satellite_adt(platDF, year_range = range(2014,2024)):
    adt_dict = {k:None for k in year_range}
    for open_yr in adt_dict.keys():
        folder = '/Volumes/crusoe-repo/data/copernicusmarine/' # used to be in OneDrive/Code/CRUSOE
        filepath = ('cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_adt-sla_179.94W-179.94E_88.94S-35.06S_' + str(open_yr) + '-01-01-' + str(open_yr) + '-12-31.nc')
        adt_dict[open_yr] = xr.open_dataset(folder + filepath)

    platDF =  mod_ocean.expand_datetime(platDF, type='dataframe')
    platDF_annual = {k:df for k, df in platDF.groupby('year')}
    for yr in year_range:
        adt_data = adt_dict[yr]
        temp = platDF_annual[yr].copy()
        temp['adt'] =  temp.apply(lambda row: get_row_adt(row, adt_data), axis=1)
        platDF_annual[yr] = temp
    platDF_added = pd.concat(platDF_annual.values(), axis=0)

    return platDF_added

def get_row_adt(row, adt_year_data):
    return adt_year_data.sel(time=row.datetime, latitude=row.latitude, longitude=row.longitude, 
                        method='nearest').adt.values.tolist()


# %% ERA5 wind speed


def get_row_windspeed(row, wind_year_data):
    nearest = wind_year_data.sel(longitude=row.longitude, latitude=row.latitude, valid_time = row.datetime,
                        method='nearest')
    return (nearest['u10'].values, nearest['v10'].values)


def add_wind_speed(platDF, year_range = range(2014,2025)): 
    wind_dict = {k:None for k in [i for i in year_range]}
    for yr in year_range:
        # yearly_era5 = xr.open_dataset('../climatedatastore/era5_winds_' + str(yr) + '.nc')
        yearly_era5 = xr.open_dataset('/Volumes/ocean-repo/era5_wind_speed_' + str(yr) + '.nc')
        
        wind_dict[yr] = yearly_era5
    
    platDF =  mod_ocean.expand_datetime(platDF, type='dataframe') # shipDF
    platDF_annual = {k:df for k, df in platDF.groupby('year')}
    for yr in tqdm(year_range):
        temp = platDF_annual[yr].copy()
        temp['wind_components'] =  temp.apply(lambda row: get_row_windspeed(row, wind_dict[yr]), axis=1)
        temp['wind_speed'] = temp['wind_components'].apply(lambda x: np.sqrt(x[0]**2 + x[1]**2))
        platDF_annual[yr] = temp
    platDF_added = pd.concat(platDF_annual.values(), axis=0)

    return platDF_added

# %% SOLAR RADIATION


from solarpy import irradiance_on_plane

def get_solar_radiation(row):
    vnorm = np.array([0, 0, -1])  # plane pointing zenith
    h = 0  # sea-level
    return irradiance_on_plane(vnorm, h, row['datetime'], row['latitude'])

def get_max_solar_radiation(row):
    vnorm = np.array([0, 0, -1])  # plane pointing zenith
    h = 0  # sea-level

    list = []
    for buffer in range(-12,13):
        time = row['datetime'] + timedelta(hours=buffer)
        list.append(irradiance_on_plane(vnorm, h, time, row['latitude']))

    return max(list)

def add_solar_radiation(platDF, daily_max=False):
    # platDF['datetime'] = pd.to_datetime(platDF['datetime'], format='%Y-%m-%d %H:%M:%S')
    platDF['datetime'] = platDF['datetime'].astype('datetime64[ns]')
    if daily_max:
        platDF['max_solar_rad'] = platDF.apply(get_max_solar_radiation, axis=1)
    else:
        platDF['solar_rad'] = platDF.apply(get_solar_radiation, axis=1)

    return platDF



# %% CDR NSIDC SEA ICE


def get_nearest_seaice(row, seaice_year_data, transformer):
    # transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3412", always_xy=True)
    x_meters, y_meters = transformer.transform(row.longitude, row.latitude)
    nearest_value = seaice_year_data.sel(x=x_meters, y=y_meters, time = row.datetime, method='nearest')['cdr_seaice_conc'].values
    return nearest_value


def add_sea_ice(platDF, year_range = range(2014,2025)): 
    """ NSIDC CDR sea ice, 25km resolution, daily """
    ice_dict = {k:None for k in [i for i in year_range]}
    for yr in year_range:
        yearly_ice = xr.open_dataset('/Volumes/crusoe-repo/data/nsidc-seaice/sic_pss25_' + str(yr) + '0101-' + str(yr) + '1231_v06r00.nc')
        ice_dict[yr] = yearly_ice
    
    platDF =  mod_ocean.expand_datetime(platDF, type='dataframe') # shipDF
    platDF_annual = {k:df for k, df in platDF.groupby('year')}

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3412", always_xy=True)

    for yr in tqdm(year_range):
        temp = platDF_annual[yr].copy()
        temp['sea_ice'] =  temp.apply(lambda row: get_nearest_seaice(row, ice_dict[yr], transformer), axis=1)
        platDF_annual[yr] = temp

    platDF_added = pd.concat(platDF_annual.values(), axis=0)
    return platDF_added



# 
# OLD
# def process_soccom_20m_clusters(bgcDS, use_var='pCO2_pHbias5_pK1', start_date = '2013-12-31', end_date = '2023-12-31'): #, use_cols=None):
#     """ 
#     Select which variable (carbon correction) to use from the 20m averages
#     Fills in missing clusters
#     Uses NCEP sea level pressure to assign atmospheric pressure (in Pa and in uatm)
    
#     """
#     soccom_df = loader.import_soccom_20m_averages(processed=False, start_date=start_date, end_date=end_date)

#     # from core clustered data
#     bgcClasses = bgcDS.to_dataframe().reset_index()
#     bgcClasses['prof_datetag'] = bgcClasses['profid'].apply(lambda x: str(x).split('_')[0])
#     bgcClasses['prof_datetag'] = bgcClasses.apply(lambda row: row.prof_datetag + '_' + row.datetime.strftime('%Y%m%d'), axis=1)
#     bgcClasses_index = bgcClasses.reset_index().groupby('prof_datetag').first()
#     bgcClasses_index

#     # Assign clusters by comparison to bgcINDEX classes
#     soccom_df['cluster'] = soccom_df['prof_datetag'].apply(lambda x: bgcClasses_index.loc[str(x)]['cluster'] if x in bgcClasses_index.index.tolist() else np.nan)

#     # Assign profid that corresponds to the bgcINDEX using the WMO number and date (prof_datetag)
#     soccom_df['profid'] = soccom_df['prof_datetag'].apply(lambda x: bgcClasses_index.loc[str(x)]['profid'] if x in bgcClasses_index.index.tolist() else np.nan)

#     print('Clusters assigned.')

#     # Fill in missing  cluster values where we can -- 
#     # look for same float, profiles within +/- 11 days
#     # If cluster was same before and after, assign to that cluster
#     soccom_df_filled = soccom_df.set_index('prof_datetag')
#     missing = soccom_df[soccom_df.cluster.isna()]
#     clustered_datetags = bgcClasses_index.reset_index().prof_datetag.unique()
#     print('Initially missing clusters for ' + str(len(missing)) + ' profiles, filling in ==>')

#     profsReplaced = []
#     # for ind in range(len(missing)):
#     for ind, tag in enumerate(missing.prof_datetag.unique()):
#         obs = missing.iloc[ind, :]
#         inds = [x for x in clustered_datetags if x.startswith(obs.wmoid)]
#         findDF = soccom_df.reset_index()
#         same_float = findDF[findDF.prof_datetag.isin(inds)].copy()

#         if len(same_float)>0:
#             same_float.loc[:,'timediff'] = same_float.apply(lambda row: abs((row.datetime - obs.datetime).days), axis=1)
#             valid = same_float[same_float.timediff<12]
#             if valid.cluster.nunique() == 1:
#                 soccom_df_filled.loc[tag, 'cluster'] = valid.cluster.values[0]
#                 profsReplaced = profsReplaced + [tag]

#     print('Filled in ' + str(len(profsReplaced)) + ' missing clusters using adjacent profiles')
#     soccom_df_filled = soccom_df_filled.rename(columns={'Lat':'latitude', 'Lon':'longitude',
#                         'MLD':'mld', 'Absolute_Salinity':'SA', 'ConservTemp':'CT'})

#     # Add sea level pressure 
#     soccom_df_filled['lon_round'] = round((soccom_df_filled['longitude'] + 180)/2.5)*2.5
#     soccom_df_filled['lat_round'] = round(soccom_df_filled['latitude']/2.5)*2.5
#     soccom_df_filled['atmos_pres_Pa'] = soccom_df_filled.apply(lambda row: get_row_sealevelpressure(row), axis=1)
#     soccom_df_filled['atmos_pres_atm'] = soccom_df_filled['atmos_pres_Pa'] / 1013.25 / 100

#     return soccom_df_filled


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
