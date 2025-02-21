

# import packages
import numpy                 as np
import xarray                as xr
import random
from   sklearn               import preprocessing
from   sklearn.decomposition import PCA
from   sklearn.decomposition import KernelPCA
from   sklearn               import manifold
from   xgcm                  import Grid



# %% Preprocessing steps 
def scale_byPressure(argoDS, vars_list= ['CT', 'SA']) -> xr.Dataset:
    """ 
    Scale variables before PCA to have mean=0 and std=1 grouped by pressure
    (means/std calculated separately for each pressure level) 
    @param: argoDS: xr dataset with dimensions of profile, pressure
                     data vars should already include CT, SA
            vars_list: list of variables to scale
    @return: Dataset with scaled variables, same dims/coords as argoDS
    """
    scaled_vars = [] # List of scaled Datasets, one for each variable in vars_list

    for var in vars_list: 
        # Initialize list of scaled DataArrays (single variable), one for each pressure level
        scaled_byPressure = [] 
        for key, group in argoDS.groupby('pressure'): # This removes the pressure dimension
            arr = xr.DataArray(preprocessing.scale(group[var], with_mean=True, with_std=True), 
                                        dims=group.dims, coords=group.coords)
            # Add scaled DataArray to list, and make pressure a dimension again
            scaled_byPressure.append(arr.expand_dims('pressure'))

        # Combine into Dataset representing one variable with "profid", "pressure" as dimensions
        scaled_vars.append(xr.concat(scaled_byPressure, dim='pressure').rename(var))

    # Combine all scaled variables into one Dataset
    scaled_DS = xr.merge(scaled_vars)

    return scaled_DS


# %% Fit the PCA model 


def train_pca(argoDS, vars_list = ['CT', 'SA'], n_components = 3, train_frac=0.333):
    """ """

    Xscaled = scale_byPressure(argoDS, vars_list = vars_list)




# def fit_and_apply_pca(profiles, variables_to_include, number_of_pca_components=3, kernel=False, train_frac=0.33, method='onZ'):
#     """ 
#     Originally from Dani Jones
#     Feb 2025 - Modified by @song-sangmin
#     """

#     Xscaled = scale_byPressure(profiles, variables_to_include)

#     # create PCA object
#     if kernel==True:
#         # KernelPCA approach (crashses due to memory)
#         print('load_and_preprocess: apply KernelPCA')
#         pca = KernelPCA(n_components=number_of_pca_components,
#                         kernel='linear', fit_inverse_transform=True, gamma=10)
#     else:
#         pca = PCA(number_of_pca_components)

#     # random sample for training
#     pf           = profiles.profile
#     rsample_size = np.min((int(train_frac*pf.size),int(pf.size)))
#     rows_id      = random.sample(range(0,pf.size), rsample_size)
#     Xtrain       = Xscaled[rows_id,:]

#     # fit PCA model using training dataset
#     print('Fitting PCA')
#     pca.fit(Xtrain)

#     # transform entire input dataset into PCA representation
#     Xpca = pca.transform(Xscaled)

#     # calculated total variance explained
#     if kernel==False:
#         total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
#         print(total_variance_explained_)

#     return pca, Xpca


# def apply_pca(profiles, pca, method='onZ'):
#     """ 
#     Originally from Hannah Joy-Warren 
#     Feb 2025 - Modified by @song-sangmin
#     """

#     # start message
#     print('load_and_preprocess.apply_pca')

#     # concatenate
#     Xraw, Xscaled = apply_scaling(profiles, method=method)

#     # transform
#     Xpca = pca.transform(Xscaled)

#     # calculated total variance explained
#     total_variance_explained_ = np.sum(pca.explained_variance_ratio_)
#     print(total_variance_explained_)

#     return Xpca




