# import packages
import numpy                 as np
import xarray                as xr
import random
from   sklearn               import preprocessing
from   sklearn.decomposition import PCA
from   sklearn.decomposition import KernelPCA
from   sklearn               import manifold
# from   xgcm                  import Grid
import pandas                as pd
from sklearn.mixture import GaussianMixture as GMM

# %% Preprocessing steps 
def center_byPressure(argoDS, vars_list= ['CT', 'SA'], with_scaling = False) -> xr.Dataset:
    """ 
    Center and optionally scale variables before PCA to have mean=0 (and std=1) grouped by pressure
    (means/std calculated separately for each pressure level) 
    @param: argoDS: xr dataset with dimensions of profile, pressure
                     data vars should already include CT, SA
            vars_list: list of variables to center/scale
            with_scaling: True if variables should be scaled to std=1, False if only centered
    @return: Dataset with centered/scaled variables, same dims/coords as argoDS
    """
    centered_vars = [] # List of centered Datasets, one for each variable in vars_list

    for var in vars_list: 
        # Initialize list of centered DataArrays (single variable), one for each pressure level
        centered_byPressure = [] 
        for key, group in argoDS.groupby('pressure'): # This removes the pressure dimension
            arr = xr.DataArray(preprocessing.scale(group[var], with_mean=True, 
                                                   with_std = with_scaling),  # Scaling optional here
                                dims=group.dims, coords=group.coords)
            # Add centered DataArray to list, and make pressure a dimension again
            centered_byPressure.append(arr.expand_dims('pressure'))

        # Combine into Dataset representing one variable with "profid", "pressure" as dimensions
        centered_vars.append(xr.concat(centered_byPressure, dim='pressure').rename(var))

    # Combine all centered variables into one Dataset
    centered_DS = xr.merge(centered_vars)

    centered_DS = centered_DS.assign_attrs(argoDS.attrs)
    # centered_DS = centered_DS.assign_attrs({'processing': 'centered by pressure to have mean=0 and std=1 for each pressure level'})

    return centered_DS

def pca_transform(Xtrain, n_comp=2, show=True):
    """
    Fit and apply PCA to input data
    @param:  Xtrain:     Dataframe of Argo data to transform (indexed by profid)
                            (M, N) where M = # profiles, N = # pressure levels x 2 
             n_comp:         number of PCA components
    @return: Xtrans:        Dataframe of PCA transformed data
             mypca:          PCA object 
             comps:          dictionary of PCA components (eigenvectors length N)
    """
    # if show: 
        # print('===> Training PCA:')
    # Fit PCA 
    mypca = PCA(n_comp).fit(Xtrain) # PCA object
    labels = ['PC'+str(x+1) for x in range(n_comp)]
    comps = {k:v for k, v in zip(labels, mypca.components_)} # PCA components (eigenvectors)

    variance_total = np.sum(mypca.explained_variance_ratio_)
    if show:
        print('n_PCs:', str(n_comp), '\texplained_variance:', str(variance_total))

    # Apply transformation to input data
    Xtrans = pd.DataFrame(mypca.transform(Xtrain), columns=comps.keys()) # Xtrans

    return Xtrans, mypca, comps

# %% Fit the PCA model 

def run_pca_and_gmm(centered_coreDS, coreDS, coreINDEX, n_pca, n_gmm, dbar_limit=1001):
    # centered_coreDS = center_byPressure(coreDS, with_scaling = False)
    # Stack temperature and salinity across pressure levels
    # centered_data = centered_coreDS

    centered_data = centered_coreDS.where(centered_coreDS.pressure < dbar_limit, drop=True)
    temp_df = centered_data.CT.transpose('profid', 'pressure').to_pandas()
    sal_df  = centered_data.SA.transpose('profid', 'pressure').to_pandas()

    # Rename columns to distinguish them
    plevels = centered_data.pressure.values
    temp_df.columns = [f'CT_{i}' for i in plevels]
    sal_df.columns  = [f'SA_{i}' for i in plevels]

    # Concatenate along columns
    pca_input = (pd.concat([temp_df, sal_df], axis=1)
                    # .drop(columns=['CT_0', 'SA_0']) 
                    .dropna(axis=0) 
                    ) 

    # ====== Scale temperature and salinity columns by their standard deviation
    temp_cols = [col for col in pca_input.columns if col.startswith("CT_")]
    temp_std = pca_input[temp_cols].stack().std()
    pca_input[temp_cols] = pca_input[temp_cols] / temp_std

    # ==== Salinity
    sal_cols = [col for col in pca_input.columns if col.startswith("SA_")]
    sal_std = pca_input[sal_cols].stack().std()
    pca_input[sal_cols] = pca_input[sal_cols] / sal_std

    # Run and store components
    [Xtrans, pca_obj, PCdict]= pca_transform(pca_input, n_comp=n_pca)
    # Xtrans['profid']=Xtrain.index.values

    # =======  Fit GMM ========
    gmm = GMM(
            n_components=n_gmm, covariance_type='diag', max_iter=10000, random_state=0,
            means_init=None)
    gmm.fit(Xtrans)


    Y_gmm = (pd.concat([Xtrans,
                    pd.Series(gmm.predict(Xtrans), name='class')], axis=1))
    Y_gmm.set_index(pca_input.index, inplace=True) # set profid as index

    # Use profids to separate out data by class
    class_locs = {k:None for k in range(gmm.n_components)}
    class_data = {k:None for k in range(gmm.n_components)}
    for ind in range(gmm.n_components):
        class_locs[ind] = coreINDEX.sel(profid=[x for x in Y_gmm[Y_gmm['class']==ind].index.values])
        class_data[ind] = coreDS.sel(profid=[x for x in Y_gmm[Y_gmm['class']==ind].index.values])

    # probs = pd.DataFrame(gmm.predict_proba(Xtrans))

    # == Print number of profiles by class
    print('Number of profiles by class')
    for k, v in class_locs.items():
        print('class', k+1, ':', len(v.profid.values))
    # len(class_locs[0].profid.values)

    return pca_input, Xtrans, pca_obj, PCdict, gmm, Y_gmm, class_locs, class_data

def calc_postprobs(gmm, Xtrain, Xtrans, Y_gmm, class_locs):
    """ 
    """
    allprobs = pd.DataFrame(gmm.predict_proba(Xtrans))
    allprobs.set_index(Xtrain.index, inplace=True) 

    postprobs = {k:None for k in range(gmm.n_components)}
    for ind in range(gmm.n_components):
        postprobs[ind] = allprobs.loc[[x for x in Y_gmm[Y_gmm['class']==ind].index.values]]

    # Make dictionary of probabilities with locations for plotting
    class_probs = {k:None for k in range(gmm.n_components)}

    for ind in range(gmm.n_components):
        temp = class_locs[ind].to_dataframe()
        temp['probability'] = postprobs[ind][ind].values
        class_probs[ind] = temp
    
    return allprobs, class_probs


# %% 
#####################################################################
# Utilities for model selection (e.g. BIC, AIC)
#####################################################################

# import modules
from sklearn import mixture
import numpy as np
import random

#####################################################################
# Calculate BIC and AIC
#####################################################################
def calc_bic_and_aic(Xpca, max_N, max_iter=20):
# calc_bic_and_aic(Xpca, max_N, max_iter=20)
# returns bic_mean, bic_std, aic_mean, aic_std

    # start message
    print('bic_and_aic.calc_bic_and_aic')
    print('--- this may take some time ---')

    # initialize, declare variables
    bic_scores = np.zeros((2,max_iter))
    aic_scores = np.zeros((2,max_iter))

    # loop through the maximum number of classes, estimate BIC
    n_components_range = range(2, max_N)
    iter_range = range(0,max_iter)

    # iterate through all the covariance types (just 'full' for now)
    cv_types = ['full']

    # loop through cv_types, components, and iterations
    for cv_type in cv_types:
        # iterate over all the possible numbers of components
        for n_components in n_components_range:
            bic_one = []
            aic_one = []
            # repeat the BIC step for better statistics
            for bic_iter in iter_range:
                # select a new random subset
                rows_id = random.sample(range(0,Xpca.shape[0]-1), 1000)
                Xpca_for_BIC = Xpca[rows_id,:]
                # fit a Gaussian mixture model
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type,
                                              random_state=42)

                # uncomment for 'rapid' BIC fitting
                gmm.fit(Xpca_for_BIC)

                # append this BIC score to the list
                bic_one.append(gmm.bic(Xpca_for_BIC))
                aic_one.append(gmm.aic(Xpca_for_BIC))
                Xpca_for_BIC = []
                Xpca_for_AIC = []

            # stack the bic scores into a single 2D structure
            bic_scores = np.vstack((bic_scores, np.asarray(bic_one)))
            aic_scores = np.vstack((aic_scores, np.asarray(aic_one)))

    # the first two rows are not needed; they were only placeholders
    bic_scores = bic_scores[2:,:]
    aic_scores = aic_scores[2:,:]

    # mean values for BIC and AIC
    bic_mean = np.mean(bic_scores, axis=1)
    aic_mean = np.mean(aic_scores, axis=1)

    # standard deviation for BIC and AIC
    bic_std = np.std(bic_scores, axis=1)
    aic_std = np.std(aic_scores, axis=1)

    return bic_mean, bic_std, aic_mean, aic_std



