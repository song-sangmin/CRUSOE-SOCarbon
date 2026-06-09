
# import packages
import numpy                 as np
import xarray                as xr
import pandas                as pd
import random
from   sklearn               import preprocessing
from   sklearn.decomposition import PCA
from   sklearn.decomposition import KernelPCA
from   sklearn               import manifold

# %% 
coreDS = xr.open_dataset('/Users/sangminsong/Library/CloudStorage/OneDrive-UW/Code/CREMAS/data/coreDS_jan2014_20250204.nc')
coreDS_scaled = coreDS.copy()


# %%


# def scale_DataArray(da):
CT_list = []
SA_list = []
# Make a dataframe that will be indexed by pressure
# output = pd.DataFrame(index='pressure')

for key, group in coreDS.groupby('pressure'):
    # Xraw = group.CT # this is a DataArray
    # Xscaled = preprocessing.scale(Xraw)
    CT_list.append(xr.DataArray(preprocessing.scale(group.CT), dims=group.dims, coords=group.coords).expand_dims('pressure'))
    SA_list.append(xr.DataArray(preprocessing.scale(group.SA), dims=group.dims, coords=group.coords).expand_dims('pressure'))

CT_scaled = xr.concat(CT_list, dim='pressure').rename('CT')
SA_scaled = xr.concat(SA_list, dim='pressure').rename('SA')
coreDS_scaled = xr.merge([CT_scaled, SA_scaled])


    # print(Xraw)
    # temp{key} = xr.DataArray(Xscaled, dims=Xraw.dims, coords=Xraw.coords)





# %%

vars_list = ['CT', 'SA']

def scale_Dataset_vars(argoDS, vars_list= ['CT', 'SA']):
    """ 
    Scale variables CT and SA to have mean=0 and std=1, grouped by pressure. 
    Uses sklearn preprocessing.scale function. 
    @param: argoDS - xr dataset with dimensions of profid, pressure
                     data vars must include CT, SA
    @return: xr dataset with scaled variables in vars_list, same dims/coords as original
    """
    scaled_vars = [] # list of scaled Datasets, one for each variable in vars_list


    for var in vars_list: 
        scaled_byPressure = [] # list of scaled DataArrays, one for each pressure level
        for key, group in coreDS.groupby('pressure'):
            scaled_array = xr.DataArray(preprocessing.scale(group[var], with_mean=True, with_std=True), 
                                        dims=group.dims, coords=group.coords)
            scaled_byPressure.append(scaled_array.expand_dims('pressure'))

        scaled_vars.append(xr.concat(scaled_byPressure, dim='pressure').rename(var))

    return xr.merge(scaled_vars)
