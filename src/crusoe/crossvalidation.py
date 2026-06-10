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
def subset_training_validation(platDF, indexer = 'wmoid', split_frac = 0.8):
    """  
    platDF: dataframe for either floatDF or shipDF
    indexer: choose whether to split by 'wmoid' or 'cruiseid' 
    """
    # Create list of unique profile ID's and select random 80% for training
    ids = platDF[indexer].unique()
    np.random.seed(42) 
    training_ids = np.random.choice(ids, int(np.floor(split_frac*len(ids))), replace=False)
    training_data = platDF[platDF[indexer].isin(training_ids)]

    validation_ids = [x for x in ids if x not in training_ids]
    validation_data = platDF[platDF[indexer].isin(validation_ids)]

    return training_data, validation_data, training_ids

def subset_folds(platDF, indexer, nfolds):
    """ 
    spatial clustering of observations for cross-validation (k-means)
    returns dictionary of training and validation dataframes for each fold
    
    @param:     platDF: dataframe for either floatDF or shipDF
                
    @return     fold_training: dictionary with keys 'fold1', 'fold2', ... each containing training dataframe
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

def subset_spatial_folds(platDF, nfolds):
    """ returns dictionary of training and validation dataframes for each fold
    (note "indexer" no longer relevant, should run on combined ship and float platforms)
    #param:     platDF: dataframe with observations
    @return     fold_training: dictionary with keys 'fold1', 'fold2', ... each containing training pd Dataframe
    """
    fold_training = {('fold'+str(k+1)):None for k in range(nfolds)}
    fold_validation = {('fold'+str(k+1)):None for k in range(nfolds)}

    kmeans = KMeans(n_clusters=nfolds, random_state=42)
    platDF['sin_longitude'] = np.sin(np.radians(platDF['longitude']))
    platDF['cos_longitude'] = np.cos(np.radians(platDF['longitude']))
    platDF['scaled_latitude'] = -1 + 2*((platDF['latitude'] - platDF['latitude'].min()) / platDF['latitude'].max()) #range -1 to 1
    platDF['fold'] = kmeans.fit_predict(platDF[['scaled_latitude', 'sin_longitude', 'cos_longitude']])

    for fnum in range(nfolds):
        fold_validation['fold' + str(fnum+1)] = platDF[platDF['fold'] == fnum].drop(columns=['fold'])
        fold_training[f'fold{fnum+1}'] = platDF[platDF['fold'] != fnum].drop(columns=['fold'])
    
    return fold_training, fold_validation


# %% Objects to store cross-validation data
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
        self.trainDF = {fold: None for fold in fold_list} # Collapsed across classes, analogous to trainDF_all in CrossValContainer
        self.valDF = {fold: None for fold in fold_list}

        # Later, populate this with other NestedCrossValContainer objects for the inner loop of tuning hyperparameters
        self.inner_nests = {fold:None for fold in fold_list} # ['fold' + str(k) for k in range(1, n_outer+1)]} 

    
    def copy(self):
        return self

class NestedModelVersion:
    """ 
    Object holding nested CV model runs
    @ param     fold_list: list of folds, inner or outer
                
                models - scikit learn RandomForestRegressor objects

                DF_err - validation DataFrame with estimation errors
    """
    def __init__(self, fold_list, feat_list=None):
        self.fold_list = fold_list
        self.feat_list = feat_list
        self.models = {fold: None for fold in fold_list}
        self.estimates = {fold: None for fold in fold_list}
        # self.calibrated_validation = None
    
    def copy(self):
        return self


