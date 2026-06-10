# moving from mod_regression to crossvalidation
import pandas as pd
import xarray as xr
import numpy as np
import scipy
from scipy.stats import iqr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# %% Data splitting functions for training and validation sets

def align_platform_indexers(plat_data : list[pd.DataFrame], 
                            plat_indexers : list[str]) -> pd.DataFrame:
    """ 
    Renames unique platform indexers (e.g. wmoid for floats) to standard "platform_id")

    :param plat_data: list of dataframes for each platform (e.g. floatDF, socatDF)
    :param plat_indexers: list of column names to use as indexers for fold splitting
                        ('wmoid' or 'cruiseid')
    
    :return None
    """
    output = []
    for ind, platDF in enumerate(plat_data): # floatDF, socatDF
        current_indexer = plat_indexers[ind]
        renamed_platDF = platDF.rename(columns={current_indexer: 'platform_id'})
        renamed_platDF['platform_indexer'] = np.tile(current_indexer, len(platDF)) # add column to keep track of original indexer type
        output.append(renamed_platDF)

    return pd.concat(output, axis=0)


def subset_training_validation(platDF, indexer = 'wmoid', split_frac = 0.8):
    """  
    Data subsetting for simple holdout validation;
    withholds a fraction of the unique platform IDS for validation
    and uses rest to train a single model.

    :param platDF: dataframe for either floatDF or shipDF
    :param indexer: column whether to split by 'wmoid' or 'cruiseid' 
    :param split_frac: fraction of unique platforms to use for training (default 0.8)
    
    :return: trainingDF: data for training, approx. 80% by default
            validationDF: data to withhold for validation
    """
    # Create list of unique profile ID's and select random 80% for training
    ids = platDF[indexer].unique()
    np.random.seed(42) 
    training_ids = np.random.choice(ids, int(np.floor(split_frac*len(ids))), replace=False)
    trainingDF = platDF[platDF[indexer].isin(training_ids)]

    validation_ids = [x for x in ids if x not in training_ids]
    validationDF = platDF[platDF[indexer].isin(validation_ids)]

    return trainingDF, validationDF, training_ids

def subset_folds(platDF, indexer='platform_id', nfolds=5) -> list[dict[str, pd.DataFrame]]:
    """ 
    Returns dictionary of training and validation dataframes for each fold.
    
    :param platDF: dataframe for either floatDF or shipDF
    :param indexer: choose whether to split by 'wmoid' or 'cruiseid'
                    default "platform_id" for combined dataset (use align_platform_indexers())
    :param nfolds: (int) number of folds for cross-validation

    :return: fold_training: dictionary with keys 'fold1', 'fold2', ... with training DF
             fold_validation: dictionary with keys 'fold1', 'fold2', ... with validation DF

    """
    # Create list of unique profile ID's and select random 80% for training
    ids = platDF[indexer].unique()
    np.random.shuffle(ids)

    holdout_ids = np.array_split(ids, nfolds) # each split is ~1/n of the data 
    fold_training = {('fold'+str(k+1)):None for k in range(nfolds)}
    fold_validation = {('fold'+str(k+1)):None for k in range(nfolds)}

    # withhold 1/n of data for validation each time, use rest for training
    for k in range(nfolds):
        fold_training[('fold'+str(k+1))] = platDF[~platDF[indexer].isin(holdout_ids[k])]
        fold_validation[('fold'+str(k+1))] = platDF[platDF[indexer].isin(holdout_ids[k])] 
    
    return fold_training, fold_validation

def subset_spatial_folds(platDF, nfolds, scale_latitude = 1) -> list[dict[str, pd.DataFrame]]:
    """ 
    Returns dictionary of training and validation dataframes for each fold
    (note "indexer" not used as parameter; works on combined platform data).



    :param platDF: dataframe of observations with columns 'latitude' and 'longitude'
    :param nfolds: (int) number of folds for cross-validation
    :param scale_latitude: float, scaling ratio for latitude/longitude in k-means
                            default of 1 gives range [-1.1] to match sinusoidal longitude range [-1,1]
                            changing to >1 gives more weight to latitude in clustering (ex: 3 yields range [-3,3])

    :return: fold_training: dictionary with keys 'fold1', 'fold2', ... with training DF
             fold_validation: dictionary with keys 'fold1', 'fold2', ... with validation DF

    """
    fold_training = {('fold'+str(k+1)):None for k in range(nfolds)}
    fold_validation = {('fold'+str(k+1)):None for k in range(nfolds)}

    kmeans = KMeans(n_clusters=nfolds, random_state=42)
    platDF['sin_longitude'] = np.sin(np.radians(platDF['longitude']))
    platDF['cos_longitude'] = np.cos(np.radians(platDF['longitude']))

    latitude_0to1 = (platDF['latitude'] - platDF['latitude'].min()) / platDF['latitude'].max() #range 0 to 1
    platDF['scaled_latitude'] = -scale_latitude + scale_latitude*2*(latitude_0to1) 
    platDF['fold'] = kmeans.fit_predict(platDF[['scaled_latitude', 'sin_longitude', 'cos_longitude']])

    for fnum in range(nfolds):
        fold_validation['fold' + str(fnum+1)] = platDF[platDF['fold'] == fnum].drop(columns=['fold'])
        fold_training[f'fold{fnum+1}'] = platDF[platDF['fold'] != fnum].drop(columns=['fold'])
    
    return fold_training, fold_validation


# %% Objects to store cross-validation data

# class ClusteredCrossValContainer:
#     """ Version that assumes n clusters, used for simple k-fold validation
#     Previously mod_reg.CrossValContainer
#     """
#     def __init__(self, fold_list, n_clusters):
#         self.fold_list = fold_list # ['fold1', ...

#         self.trainClasses = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
#         self.valClasses = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}

#         # self.trainClasses_float = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
#         # self.trainClasses_ship = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
#         # self.valClasses_float = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
#         # self.valClasses_ship = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}

#         self.trainDF_all = {fold: None for fold in fold_list}
#         self.valDF_all = {fold: None for fold in fold_list}
#         self.countobs = {fold: None for fold in fold_list}
    
#     def copy(self):
#         return self

# class NestedModelVersion:
#     """ 
#     Object holding nested CV model runs
#     @ param     fold_list: list of folds, inner or outer
                
#                 models - scikit learn RandomForestRegressor objects

#                 DF_err - validation DataFrame with estimation errors
#     """
#     def __init__(self, fold_list, feat_list=None):
#         self.fold_list = fold_list
#         self.feat_list = feat_list
#         self.models = {fold: None for fold in fold_list}
#         self.estimates = {fold: None for fold in fold_list}
#         # self.calibrated_validation = None
    
#     def copy(self):
#         return self



class NestedCrossValContainer: 
    """ Nested cross-validation version (Schratz 2019) that has an outer loop and inner loop for hyperparameter tuning
    """
    def __init__(self, nfolds=5):
        """ 
        @ param     nfolds: number of folds (outer or inner)
        """
        fold_list = ['fold' + str(k) for k in range(1, nfolds+1)]

        # Outer folds for main model comparison
        self.fold_list = fold_list

        # rename from trainDF, valDF
        self.training_data = {fold: None for fold in fold_list} # Collapsed across classes, analogous to trainDF_all in CrossValContainer
        self.validation_data = {fold: None for fold in fold_list}

        # Later, populate this with other NestedCrossValContainer objects for the inner loop of tuning hyperparameters
        self.inner_nests = {fold:None for fold in fold_list} # ['fold' + str(k) for k in range(1, n_outer+1)]} 

    
    def copy(self):
        return self


# %% Create cross-validation containers

def create_cv_nest(trainval_data,
                    n_split: int,
                    spatial_cv = False,
                    scale_latitude = 1,
                    indexer='platform_id'): 
    """ 
    Internal function in setup_cv_container.
    Equivalent to calling setup_cv_container with n_inner = None

    :param trainval_data: dataframe with data to split into training/validation folds
    :param n_splits
    :param spatial_cv: whether to use spatial clustering for fold splits 
                    default False, i.e. random splits by platform)
                    if 'k-means', use clustering on lat/lon coordinates
    :param scale_latitude: float, scaling ratio for latitude/longitude in k-means
                    default of 1 gives range [-1.1] to match sinusoidal longitude range [-1,1]
                    changing to >1 gives more weight to latitude in clustering (ex: 3 yields range [-3,3])

    :param indexer: column name to use for identifying unique platforms for splitting 
                    default 'platform_id' if running .align_platform_indexers() on combined dataset

    """
    # Set up folds for cross-validation, with optional nesting
    cvtainer = NestedCrossValContainer(n_split) # Store in soclass_cvtainers[soclass]
    if spatial_cv == False:
        [cvtainer.training_data, cvtainer.validation_data] = subset_folds(trainval_data, 
                                                                            indexer=indexer, 
                                                                            nfolds=n_split)
    elif spatial_cv == 'k-means':
        [cvtainer.training_data, cvtainer.validation_data] = subset_spatial_folds(trainval_data,
                                                                            nfolds=n_split,
                                                                            scale_latitude=1)
    return cvtainer

def setup_cv_container(trainval_data,
                       n_outer: int,
                       n_inner = None, 
                       spatial_cv = False,
                       indexer='platform_id') -> NestedCrossValContainer: 
    """ 
    Sets up nested cross-validation containers for each class

    :param trainval_data: dataframe with data to split into training/validation folds
    :param n_outer: number of outer folds; performance estimation level
    :param n_inner: number of inner folds; hyperparameter tuning level  
                    default None, equivalent to non-nested CV) 
    :param spatial_cv: whether to use spatial clustering for fold splits 
                    default False, i.e. random splits by platform)
                    if 'k-means', use clustering on lat/lon coordinates
    :param scale_latitude: float, scaling ratio for latitude/longitude in k-means
                    default of 1 gives range [-1.1] to match sinusoidal longitude range [-1,1]
                    changing to >1 gives more weight to latitude in clustering (ex: 3 yields range [-3,3])
    :param indexer: column name to use for identifying unique platforms for splitting 
                    default 'platform_id' if running .align_platform_indexers() on combined dataset

    :return: NestedCrossValContainer with outer folds for performance estimation
                        and optional inner folds for tuning
    """
    # Set up outer folds, for both ship and float
    outer_nest = create_cv_nest(trainval_data, n_outer, spatial_cv=spatial_cv, indexer=indexer)
    
    # Optional inner folds for hyperparameter tuning (nested CV)
    if n_inner is not None:
        for outer_fold_tag in outer_nest.fold_list: 
            # Use only the training data from the outer fold to make a new set of splits 
            inner_nest = create_cv_nest(outer_nest.training_data[outer_fold_tag], 
                                        n_inner, spatial_cv=spatial_cv, indexer=indexer)
            outer_nest.inner_nests[outer_fold_tag] = inner_nest
    else: outer_nest.inner_nests = None

    return outer_nest

# %% Useful functions for data diagnostics

def get_cvtainer_counts(soclass_cvtainers, n_clusters, n_outer, n_inner=None):
    """ function for getting # of observations in nested architecture
    Works with either nested or non-nested (n_inner=None) CV objects

    @param: soclass_cvtainers: dictionary of mod_reg.NestedCrossValContainers, one for each class
            n_clusters: number of southern ocean classes
            n_outer: number of outer folds
            n_inner: number of inner folds (if None, will just get outer fold counts)
    """
    arrays = [['cluster'+str(x) for x in range(1, n_clusters+1)],
       ['fold'+str(x) for x in range(1, n_outer+1)]]
    colnames = ["cluster", "outfold"]

    if n_inner is not None:
       arrays = arrays + [['total'] + ['fold' + str(x) for x in range(1, n_inner+1)]]
       colnames = colnames + ['infold']

    index = pd.MultiIndex.from_product(arrays, names=colnames)
    NestedCvtainerCounts = pd.DataFrame(index = index, columns= ['train_counts', 'val_counts'])
    for classnum in ['cluster'+str(x) for x in range(1, n_clusters+1)]:
        for foldnum in ['fold'+str(x) for x in range(1, n_outer+1)]:
           
            if n_inner is None:
                NestedCvtainerCounts.loc[(classnum, foldnum), 'train_counts'] = len(soclass_cvtainers[classnum].training_data[foldnum])
                NestedCvtainerCounts.loc[(classnum, foldnum), 'val_counts'] = len(soclass_cvtainers[classnum].validation_data[foldnum])
            
            if n_inner is not None:
                NestedCvtainerCounts.loc[(classnum, foldnum, 'total'), 'train_counts'] = len(soclass_cvtainers[classnum].training_data[foldnum])
                NestedCvtainerCounts.loc[(classnum, foldnum, 'total'), 'val_counts'] = len(soclass_cvtainers[classnum].validation_data[foldnum])
                inner_nest = soclass_cvtainers[classnum].inner_nests[foldnum] 
                for tuningnum in ['fold'+str(x) for x in range(1, n_inner+1)]:
                    NestedCvtainerCounts.loc[(classnum, foldnum, tuningnum), 'train_counts'] = len(soclass_cvtainers[classnum].inner_nests[foldnum].training_data[tuningnum])
                    NestedCvtainerCounts.loc[(classnum, foldnum, tuningnum), 'val_counts'] = len(soclass_cvtainers[classnum].inner_nests[foldnum].validation_data[tuningnum])

    return NestedCvtainerCounts