import pandas as pd
import xarray as xr
import numpy as np
import scipy
import math
from scipy.stats import iqr

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# %% SETUP  RUN TAGS

def make_feat_dict(feat_lists):
    """ 
    @param    feat_lists: list of lists of feature names for each cluster
    """
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    labels = ['feat'+ ascii_uppercase[i] for i in range(len(feat_lists))]
    feat_options = {labels[i]: feat_lists[i] for i in range(len(feat_lists))}

    return feat_options 

def make_run_tags(feat_options, data_options, target_options):
    """ 
    @ param   feat_options: dict of feature lists (make_feat_dict())
               data_options: dict of datasets for training, validation
                    data_options = {'float': [trainClasses_float, valClasses],
                        'ship': [trainClasses_ship, valClasses],
                        'combined': [trainClasses, valClasses]}


               target_options: list of target variable names"""
    # Automatically generate run tag combinations
    run_tags = []
    for key1 in feat_options.keys():
        for key2 in data_options.keys():
            for target in target_options:
                run_tags.append(key1 + '-' + key2 + '-' + target)
    return run_tags


# %% TRAINING AND VALIDATION DATA

def subset_training_validation(platDF, indexer = 'wmoid', split_frac = 0.8):
    """  
    platDF: dataframe for either floatDF or shipDF
    indexer: choose whether to split by 'wmoid' or 'cruiseid' 
    """
    # Create list of unique profile ID's and select random 80% for training
    ids = platDF[indexer].unique()

    # np.random.seed(42) 
    training_ids = np.random.choice(ids, int(np.floor(split_frac*len(ids))), replace=False)
    training_data = platDF[platDF[indexer].isin(training_ids)]

    validation_ids = [x for x in ids if x not in training_ids]
    validation_data = platDF[platDF[indexer].isin(validation_ids)]

    return training_data, validation_data, training_ids

# def subset_folds(platDF, indexer, nfolds):

def subset_folds(platDF, indexer, nfolds):
    """ returns dictionary of training and validation dataframes for each fold
    (for single platform)
    #param:     platDF: dataframe for either floatDF or shipDF
                indexer: 'cruiseid' for shipDF, 'wmoid' for floatDF
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

def get_trainval_counts(training_classes, validation_classes, n_gmm=8):
    """  
    originally from 3.1_clustered_rfr.ipynb 
    @param  training_classes: dict of training dataframes for each cluster

            validation_classes: dict of validation dataframes for each cluster"""
    countobs = pd.DataFrame({'cluster':range(1,n_gmm+1), 
                        'train':[len(k) for k in training_classes.values()],
                        'validation':[len(k) for k in validation_classes.values()]}
                         )

    countobs['train_float'] = [len(k[~k['wmoid'].isna()]) for k in training_classes.values()]
    countobs['train_ship'] = [len(k[k.wmoid.isna()]) for k in training_classes.values()]

    countobs['val_float'] = [len(k[~k['wmoid'].isna()]) for k in validation_classes.values()]
    countobs['val_ship'] = [len(k[k.wmoid.isna()]) for k in validation_classes.values()]

    # countobs['train_pc_float'] = countobs.apply(lambda row: row['train_float']/row['train']*100, axis=1)
    countobs['train_pc_float'] = countobs.apply(lambda row: row['train_float']/row['train']*100, axis=1)
    countobs['val_pc_float'] = countobs.apply(lambda row: row['val_float']/row['validation']*100, axis=1)

    countobs['TOTAL_FLOAT'] = countobs['train_float'] + countobs['val_float']
    countobs['TOTAL_SHIP'] = countobs['train_ship'] + countobs['val_ship']

    countobs['TOTAL_OBS'] = countobs['train'] + countobs['validation']

    # print('percent split into training/validation: ' + str(len(trainDF_all)/(len(trainDF_all)+len(valDF_all))*100))
    countobs.set_index('cluster', inplace=True)

    return countobs

def find_trainval_split(floatDF, shipDF, pc_thresh=0.3, n_gmm=6):
    """ 
    Main splitting function used in 3.1_clsutered_RFR
    Updated Jan 19 2026
    Instead of doing probability-based assignment, do signle assignment
    Iteratively search for a training/validation split that gives at least X% float

    @ param     pc_thresh : proportion of float data in training and validation sets
    """

    platDF_all = pd.concat([floatDF, shipDF], axis=0)
    trainClasses = {k:None for k in range(1, n_gmm+1)}
    valClasses = {k:None for k in range(1, n_gmm+1)}

    # pc_thresh = 0.3
    for soclass in range(1, n_gmm+1):
        print('==> Processing class ' + str(soclass))
        classDF = platDF_all[platDF_all['class_assign'] == soclass]
        classDF_float = classDF[~classDF.wmoid.isna()]
        classDF_ship = classDF[classDF.wmoid.isna()]

        pc_train_float = 0; pc_val_float = 0
        print('Searching for valid splits...')
        while (pc_train_float < pc_thresh) or (pc_val_float < pc_thresh):
            # Iteratively search for split until both training and validation have at least pc_thresh float
            [holdTrain_float, holdVal_float, _] = subset_training_validation(classDF_float, indexer='wmoid', split_frac=0.8)
            [holdTrain_ship, holdVal_ship, _] = subset_training_validation(classDF_ship, indexer='cruiseid', split_frac=0.8)
            # print('trying split')
            pc_train_float = len(holdTrain_float) / (len(holdTrain_float) + len(holdTrain_ship))
            pc_val_float = len(holdVal_float) / (len(holdVal_float) + len(holdVal_ship))

        trainClasses[soclass] = pd.concat([holdTrain_float, holdTrain_ship], axis=0)
        valClasses[soclass] = pd.concat([holdVal_float, holdVal_ship], axis=0)
        
    trainClasses_float = { ind : k[~k.wmoid.isna()] for ind, k in trainClasses.items()}
    trainClasses_ship = { ind : k[k.wmoid.isna()] for ind, k in trainClasses.items()}

    trainDF_all = pd.concat(trainClasses.values(), axis=0)
    valDF_all = pd.concat(valClasses.values(), axis=0)
    print('Resulting split for training/validation observations: ' + str(len(trainDF_all)/(len(trainDF_all)+len(valDF_all))*100))

    countobs = get_trainval_counts(trainClasses, valClasses, n_gmm=n_gmm)
    countobs

    return [trainClasses, valClasses, trainDF_all, valDF_all, countobs]

def separate_platforms(platDF_combined):
    """ 
    Split combined dataframe into float and ship dataframes
    @ param     platDF_combined: combined dataframe with both float and ship data
    @ return    floatDF, shipDF
    """
    
    floatDF = platDF_combined[~platDF_combined.wmoid.isna()].copy()
    shipDF = platDF_combined[~platDF_combined.cruiseid.isna()].copy()

    return [floatDF, shipDF]



# OUTDATED. moving to mod_pcm and doing exclusion before splitting 
# def exclude_classes(exclude_nums, trainClasses, n_gmm=8, reassign_numbers=True):
#     """ 
#     jan 2026 : changing to do one dict at a time
#     After splitting, remove sea ice zone 

#     # by convention, "class" is what is returned by pcm module, index starting at 0 
#     # after excluding chosen classes, reassign number to "cluster" starting at 1
#     # rename probability cols
#     @ return    exclude_nums: list of class numbers to exclude (e.g. [1, 4])

#     @ param     reassign_numbers: if True, reassign class numbers to be sequential after exclusion
    
#     """
#     valid_classnums = [k for k in range(1, n_gmm+1) if k not in exclude_nums]
#     # print(trainClasses.keys())
#     trainClasses_excluded = {}

#     for k in valid_classnums:
#         # print(valid_classnums)
#         trainClasses_excluded[k] = trainClasses[k]

#     if reassign_numbers:
#         trainClasses_excluded = reassign_class_numbers(trainClasses_excluded)

#     # Collapse into new dataframes
#     trainDF = pd.concat(trainClasses_excluded.values(), axis=0)

#     return [trainClasses_excluded, trainDF]


# moved to mod_pc
# def reassign_class_numbers(trainClasses_valid):
#     """  Works for valClasses as well"""

#     # If you excluded some, need to renumber the classes
#     trainClasses_final ={}
#     # The ind here is the new class number. Iterate over the old labels/keys
#     for ind, label in enumerate(trainClasses_valid.keys()): # label is the original class number
#         # Rename the dataframe in each item of trainClasses_valid, and save in new dictionary _final
#         new_classnum = ind + 1
#         classDF = trainClasses_valid[label].copy() # original column labels with "classK_prob"

#         # Rename probability columns + cluster column to match new class numbers
#         renamedDF = classDF[[x for x in classDF.columns if 'prob' not in x]].copy() # don't copy probabilities yet
#         for ind2, label2 in enumerate(trainClasses_valid.keys()): # 
#             renamedDF['class' + str(ind2+1) + '_prob'] = classDF['class' + str(label2) + '_prob'] 
#         renamedDF['cluster'] = np.tile(new_classnum, len(renamedDF))
        
#         # Store in new dictionary
#         trainClasses_final[new_classnum] = renamedDF
    
#     return trainClasses_final



# %% Main training function for RFR
# OLD FUNCTIONS FROM NOT CLUSTERED VERSION
# def train_RFR(feat_list, 
#               training, validation,
#               var_predict = 'delta_fco2',
#               loss_criterion = 'absolute_error',
#               cols_ignore_nans = [],
#               ntrees=1000, max_feats = 1/3, min_samples_split=5):
#     """ 
#     Main method to train the RF model.
#     Update 04.18.24: Redo bootstrapping. 
#     @param: 
#         feat_list: list of features to use in the model
#         training: training data unscaled, i.e. original range of values
#         validation: validation data unscaled
#         test: test data unscaled 
#         ntrees: 1000 trees by default.
#         loss_criterion: 'squared_error' (minimise MSE) or 'absolute_error' (MAE)

#     @return:
#         Mdl: trained RF model
#         Mdl_results: 
#         Mdl_prediction: Dataframe with error metrics for the *VALIDATION* set
#     """

#     Mdl = RandomForestRegressor(ntrees, max_features = max_feats, 
#                                 criterion = loss_criterion,
#                                 random_state = 0, bootstrap=True, 
#                                 min_samples_split=min_samples_split)
#         #  max_features: use at most X features at each split (m~sqrt(total features))

#     # Drop NaN's without profid or wmoid
#     use_cols = feat_list + [var_predict]
#     training_nona = training.dropna(axis=0, subset=use_cols)  
#     validation_nona = validation.dropna(axis=0, subset=use_cols)


#     # OLD dec 2025
#     # cols_na = [col for col in training.columns if col not in cols_ignore_nans] 
#     # training_nona = training #.dropna(axis=0, subset=cols_na)  
#     # validation_nona = validation #.dropna(axis=0, subset=cols_na)

#     # Train the model 
#     X_training = training_nona[feat_list].to_numpy()
#     Y_training = training_nona[var_predict].to_numpy()
#     Mdl.fit(X_training, Y_training)

#     # Estimate and get error predicts
#     Y_pred_validation = Mdl.predict(validation_nona[feat_list].to_numpy())
#     AE_RF_validation = np.abs(Y_pred_validation - validation_nona[var_predict])
#     IQR_RF_validation = iqr(abs(AE_RF_validation))
#     r2_RF_validation = r2_score(validation_nona[var_predict], Y_pred_validation)

#     # Validation error DF to return
#     Mdl_prediction = validation_nona.copy() #pd.DataFrame()
#     Mdl_prediction['val_prediction'] = Y_pred_validation
#     Mdl_prediction['val_error'] = Y_pred_validation - validation_nona[var_predict]
#     Mdl_prediction['val_relative_error'] = Mdl_prediction['val_error']/validation_nona[var_predict]

#     Mdl_results = [np.nanmedian(abs(AE_RF_validation)), IQR_RF_validation, r2_RF_validation]

#     return [Mdl, Mdl_results, Mdl_prediction]


# def test_RFR(feat_list, Mdl, 
#               test, 
#               var_predict = 'delta_fco2'):  
#     """ For running model once and testing on multiple synthetic test sets."""
#     # cols_na = [col for col in training.columns if col not in ['profid', 'wmoid', 'AOU', 'dist_maxb']]
#     # test_nona = test.dropna(axis=0, subset=cols_na)
#     test_nona = test.dropna(axis=0, subset=feat_list)
#     X_test = test_nona[feat_list].to_numpy()
#     Y_test = test_nona[var_predict].to_numpy()
#     Y_pred_test = Mdl.predict(X_test)

#     # Create dataframe for the test set with depth --> 
#     Mdl_test_error = test_nona.copy(); 
#     Mdl_test_error = Mdl_test_error.reset_index(drop=True)
#     observed = Mdl_test_error[var_predict].to_numpy()

#     # Save new dataframe with test results
#     Mdl_test_error['test_prediction'] = Y_pred_test
#     Mdl_test_error['test_error'] = Mdl_test_error['test_prediction'] - observed
#     Mdl_test_error['test_relative_error'] = Mdl_test_error['test_error']/observed

#     # AE_RF_test = np.abs(Y_pred_test - test_nona[var_predict]) # abs val of Mdl_test_error['test_error']
#     # IQR_RF_test = iqr(abs(AE_RF_test))
#     # r2_RF_test = r2_score(test_nona[var_predict], Y_pred_test)

#     return Mdl_test_error



# %% NEW METHODS FOR CLASS-DEPENDENT REGRESSION
# UPDATED FEB 2026

def fit_single_RFR(feat_list, 
              trainingDF, 
              var_predict = 'delta_fco2',
              loss_criterion = 'squared_error',
              ntrees=1000, max_feats = 1/3, min_samples_split=5):
    """ 
    Fit single RFR for a class. 
    Compute validation errors later 
    """
    Mdl = RandomForestRegressor(ntrees, max_features = max_feats, random_state = 0, 
                                criterion = loss_criterion,
                                bootstrap=True, min_samples_split=min_samples_split)
        #  max_features: use at most X features at each split (m~sqrt(total features))

    

    trainingDF= trainingDF.dropna(subset=feat_list).copy()

    if 'mld' in feat_list:
        trainingDF['log_mld'] = np.log(trainingDF['mld'])
        feat_list = [x if x != 'mld' else 'log_mld' for x in feat_list]

    # Train the model 
    X_training = trainingDF[feat_list].to_numpy()
    Y_training = trainingDF[var_predict].to_numpy()
    Mdl.fit(X_training, Y_training)

    return Mdl

def apply_cluster_RFR(feat_list, ClusteredModVer, sampleDF, sample_tag='val', 
                      target_var = 'delta_pco2', target_known = True): 
    """ 
    @param      feat_list list of features in sampleDF to use
                ClusteredModVer: already trained ClusteredModelVersion object
                sampleDF: dataframe with dataset to get errors on 
                sample_tag: specify a string (default "val" returns DF with cols 'val_error')
                            only need if target_known = True
    """
    sampleDF = sampleDF.dropna(axis=0, subset=feat_list).copy()
    
    if 'mld' in feat_list:
        sampleDF['log_mld'] = np.log(sampleDF['mld'])
        feat_list = [x if x != 'mld' else 'log_mld' for x in feat_list]

    k_list = ClusteredModVer.ind_list
    for k in k_list:
        pred_col = 'cluster' + str(k) + '_pred'
        sampleDF[pred_col] = ClusteredModVer.models[k].predict(sampleDF[feat_list].to_numpy())
    sampleDF['weighted_pred'] = sampleDF.apply(lambda row: weighted_prediction(row, k_list), axis=1)

    if target_known:
        sampleDF[sample_tag + '_error'] = sampleDF['weighted_pred'] - sampleDF[target_var]
        sampleDF[sample_tag + '_relative_error'] = sampleDF[sample_tag + '_error'] / sampleDF[target_var]
    
    return sampleDF
    
# %% temp xgb hold

from xgboost import XGBRegressor

# # https://xgboost.readthedocs.io/en/latest/treemethod.html
# # XGBoost: A Scalable Tree Boosting System
# # Tianqi Chen, Carlos Guestrin

# def fit_cluster_XGB(feat_list, 
#               trainingDF, 
#               var_predict = 'delta_fco2',
#               ntrees=1000, 
#               booster = 'gbtree',
#               tree_method = 'approx'):
#     """ 
#     Fit single RFR for a class. 
#     Compute validation errors later 
#     """
#     Mdl = XGBRegressor(n_estimators=ntrees, 
#                       booster = booster, 
#                       tree_method=tree_method)

#     # Train the model 
#     X_training = trainingDF[feat_list].to_numpy()
#     Y_training = trainingDF[var_predict].to_numpy()
#     Mdl.fit(X_training, Y_training)

#     return Mdl

# def run_clustered_XGB(feat_list, 
#               trainingDF, 
#               validationDF,
#               var_predict = 'delta_pco2',
#               prob_thresh = 0.3,
#               cols_ignore_nans = [],
#               ntrees=1000, booster = 'gbtree',
#               tree_method = 'approx'):
#     """ 
    
#     """
#     # Set up object to store models by SO class
    
#     k_list = range(1,9) # Number of clusters
#     training_classes = {k:None for k in k_list}

#     for k in k_list:
#         # Train regression for each class 
#         prob_col = 'cluster' + str(k) + '_prob'
#         training_classes[k] = trainingDF[trainingDF[prob_col] > prob_thresh]

#     clustered_run = myreg.ClusteredModelVersion(ind_list = k_list)

#     # === TRAINING
#     for k in tqdm(k_list): 
#         # Set training data by class
#         training_data = training_classes[k]
#         # print('Training for class ' + str(k) + ', n=' + str(len(training_data)) + '')

#         clustered_run.models[k] = fit_cluster_XGB(feat_list, 
#                                 training_data, 
#                                 var_predict = var_predict, 
#                                 ntrees = ntrees, booster = 'gbtree',
#                                 tree_method = 'approx')
#     # === VALIDATION 
#     for k in tqdm(k_list): 
#         # For each row in validationDF, apply the regression to all the data
#         pred_col = 'cluster' + str(k) + '_pred'
#         validationDF[pred_col] = clustered_run.models[k].predict(validationDF[feat_list].to_numpy())

#     # Make final predictions by weighting by class probabilities
#     def weighted_prediction(row):
#         total = 0
#         for k in k_list:
#             prob_col = 'cluster' + str(k) + '_prob'
#             pred_col = 'cluster' + str(k) + '_pred'
#             total += row[prob_col] * row[pred_col]
#         return total
    
#     validationDF['weighted_pred'] = validationDF.apply(lambda row: weighted_prediction(row), axis=1)
#     validationDF['val_error'] = validationDF['weighted_pred'] - validationDF[var_predict]
#     validationDF['val_relative_error'] = validationDF['val_error'] / validationDF[var_predict]

#     clustered_run.errorDF = validationDF

#     # === ERROR METRICS

#     total_MAE = np.round(np.nanmedian(np.abs(validationDF.val_error)),3)
#     bias = np.round(np.sum(validationDF.val_error)/len(validationDF),3)

#     validationDF['val_error_sq'] = validationDF['val_error']**2
#     mse = np.sum(validationDF['val_error_sq']) / len(validationDF.val_error)
#     rmse = np.sqrt(mse)

#     clustered_run.model_metrics = pd.DataFrame(columns=['MAE', 'bias', 'RMSE'], data=[[total_MAE, bias, rmse]])
#     print('MAE: ' + str(total_MAE) + ', bias: ' + str(bias) + ', RMSE: ' + str(rmse) + '')

#     return clustered_run

# %% Methods for preparing ML Data

# def subset_data(platDF, indexer = 'profid', split_frac = 0.8):
#     """ 
#     @param platDF
#         indexer: name of dimension to use for splitting ('profid' for Argo, 'sid' for SOCAT)
#     @return: 
#         training_data: training data (80% of profiles) 
#         validation_data: validation data (20% of profiles)

#     """
#     n_obs = int(split_frac * len(platDF))
#     ids = np.random.choice(platDF[indexer].unique(), n_obs)

#     training_data = platDF[platDF[indexer].isin(ids)].copy()
#     validation_data = platDF[~platDF[indexer].isin(ids)].copy()

#     return training_data, validation_data




# %% Classes for storing model results
class ModelVersion: 
    """ 
    Object holding different model runs
    @ param     ind_list = class indices e.g. range(1,9)
                
                models - scikit learn RandomForestRegressor objects
                errorstats - maybe get rid of. holds error summary stats
                DF_err - validation DataFrame with estimation errors
    """
    def __init__(self, ind_list):
        self.ind_list = ind_list
        self.models = {model: None for model in ind_list}
        self.errorstats = {model: None for model in ind_list}
        self.DF_err = {model: None for model in ind_list}
    
    def add(self, model_name):
        self.models[model_name] = None
        self.errorstats[model_name] = None
        self.DF_err[model_name] = None

    def copy(self):
        return self

class ClusteredModelVersion:
    """ 
    Object holding different model runs
    @ param     ind_list = class indices e.g. range(1,9)
                
                models - scikit learn RandomForestRegressor objects
                errorstats - maybe get rid of. holds error summary stats
                DF_err - validation DataFrame with estimation errors
    """
    def __init__(self, ind_list, feat_list=None):
        self.ind_list = ind_list
        self.feat_list = feat_list
        self.models = {model: None for model in ind_list}
        # self.DF_err = {model: None for model in ind_list} # validation errors from one class
        self.weighted_training = None
        self.weighted_validation = None
        # self.calibrated_validation = None
    
    def copy(self):
        return self

class CrossValContainer:
    def __init__(self, fold_list, n_clusters):
        self.fold_list = fold_list # ['fold1', ...

        self.trainClasses = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
        self.valClasses = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}

        self.trainClasses_float = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
        self.trainClasses_ship = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
        self.valClasses_float = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}
        self.valClasses_ship = {fold: {k:None for k in range(1, n_clusters+1)} for fold in fold_list}

        self.trainDF_all = {fold: None for fold in fold_list}
        self.valDF_all = {fold: None for fold in fold_list}
        self.countobs = {fold: None for fold in fold_list}
    
    def copy(self):
        return self

# Make final predictions by weighting by class probabilities
def weighted_prediction(row, ind_list, pred_col_tag = '_pred'):
    """ ind_list or k_list is range(1,n_gmm+1)

    @ param: pred_col_tag = '_pred' by default
                            for calibration, pass '_calpred'
                    
    """
    total = 0
    for k in ind_list:
        prob_col = 'cluster' + str(k) + '_prob'
        pred_col = 'cluster' + str(k) + pred_col_tag
        total += row[prob_col] * row[pred_col]
    return total



def summarize_DF_errors(platDF, error_param = 'val_error', target_var = 'delta_pco2'):
    """ 
    summarize validation errors for any Dataframe with "val_error" column

    # runResults = singleRun_collapse_errors(singleRun)
    # [run_median_abs_error, run_mean_abs_error, run_bias, run_rmse] = summarize_DF_errors(runResults)

    should move to mod_evaluation.py eventually
    """
    err = platDF[error_param]
    median_abs_error = np.abs(err).median()
    mean_abs_error = np.abs(err).mean()
    bias = (err.mean())

    platDF[error_param + '_sq'] = platDF[error_param]**2
    mse = np.sum(platDF[error_param + '_sq']) / len(platDF[error_param])
    rmse = np.sqrt(mse)

    # absolute percentage error
    # platDF['ape'] = np.abs(platDF['val_relative_error'])*100
    # median_ape = platDF['ape'].median()
    # mean_ape = platDF['ape'].mean()

    result = ([median_abs_error, mean_abs_error, bias, rmse])
    return result

# def compare_run_results(storedRuns):
def storedRuns_comparison(storedRuns, run_tags = None, error_param='val_error', 
                          target_var='delta_pco2',
                          show_platforms = ''): # float, ship, or combined
    """ 
    storedRuns: dictionary of ModelVersion objects, runtag as keys"""
    total_MAEs = pd.DataFrame()

    if run_tags is None: run_tags = [x for x in storedRuns.keys()]

    for run_tag in run_tags[:]:
        singleRun = storedRuns[run_tag]
        # print('==> Results for ' + run_tag)
        # print('\t features ', feat_options[run_tag.split('-')[0]])

        runResults = singleRun.weighted_validation.copy()
        # OUTDATED below, from before weighting predictions
        # Better method to collapse errors across classes rather than averaging the medians
        # runResults = singleRun_collapse_errors(singleRun)

        
        # runResults['val_error'] = runResults[param] - runResults[target_var]
        # runResults['val_relative_error'] = runResults['val_error'] / runResults[target_var]

        [run_median_abs_error, run_mean_abs_error, run_bias, run_rmse] = summarize_DF_errors(runResults, error_param=error_param)

        total_MAEs.loc[run_tag, 'median_AE'] = run_median_abs_error
        total_MAEs.loc[run_tag, 'mean_AE'] = run_mean_abs_error
        total_MAEs.loc[run_tag, 'bias'] = run_bias
        total_MAEs.loc[run_tag, 'RMSE'] = run_rmse

    if show_platforms in ['float', 'ship', 'combined']:
        show_rows = [x for x in run_tags if show_platforms in x]
        total_MAEs = total_MAEs.loc[show_rows, :]

    # total_MAEs['labels'] = feat_list_labels
    # total_MAEs.set_index('labels', inplace=True)
    return total_MAEs


# %% as it was before weighting error :( )
def singleRun_collapse_errors(singleRun):
    """ 
    note: used to be collapse_DF_errors() 
    Collapse DF_error from each class (within a singleRun) into single dataframe"""
    ind_list = [x for x in singleRun.ind_list]
    combined = pd.concat([singleRun.DF_err[x] for x in ind_list], axis=0)
    return combined

def singleRun_class_summary(singleRun, error_param = 'val_error'):
    """ 
    Summarize error statistics for each class, and also overall
    """
    ind_list = [x for x in singleRun.ind_list]
    full_validation = singleRun.weighted_validation.copy()
    result = pd.DataFrame(index=ind_list, columns = ['median_AE', 'mean_AE', 'bias', 'rmse'])
    for ind in ind_list:
        # result.loc[ind, :] = summarize_DF_errors(singleRun.DF_err[ind]) # old
        class_error = full_validation[full_validation['cluster'] == ind].copy()
        result.loc[ind, :] = summarize_DF_errors(class_error, error_param = error_param)

    # result.loc['overall', :] = summarize_DF_errors(singleRun_collapse_errors(singleRun))
    result.loc['overall', :] = summarize_DF_errors(full_validation, error_param = error_param)
    return result

def singleRun_platform_summary(singleRun, error_param = 'val_error'):
    """ 
    note: used to be summarize_DF()
    Summarize error statistics for a single run, split by platform type
    
    runResults is the collapsed DF_err from a ModelVersion object
    run_tag = 'featD-combined-delta_pco2'
    runResults = singleRun_collapse_errors(storedRuns_withk4[run_tag])
    """
    runResults = singleRun.weighted_validation.copy() # singleRun_collapse_errors(singleRun)

    # print('Validation errors within single run: ')
    result = pd.DataFrame(index=['median_AE', 'mean_AE', 
                                # 'median_ape', 'mean_ape',
                                'bias', 'rmse'])
    result['overall'] = summarize_DF_errors(runResults, error_param = error_param)

    [errors_float, errors_ship] = separate_platforms(runResults)
    result['float_component'] = summarize_DF_errors(errors_float)
    result['ship_component'] = summarize_DF_errors(errors_ship)

    return result.T



# def get_ModelVersion_results(singleRun, print_results=True):
#     """ 
#     the newer version of this (get errors by class in one singleRun is in ()
#     This is a way to get a dictionary 
#     singleRun is one ModelVersion object
#     singleRun = storedRuns[run_tag]
    
    
#     """
#     result_DF = pd.DataFrame(columns=['MAE', 'IQR', 'R2', 'mean_bias'])
#     total = []
#     for key, val in singleRun.errorstats.items():
#         # print('Class ' + str(key) + ': MAE=' + str(np.round(val[0],2)) + ', IQR=' + str(np.round(val[1],2)) + ', R2=' + str(np.round(val[2],2)) + '')
#         result_DF.loc[key, 'MAE'] = np.round(val[0],2)
#         result_DF.loc[key, 'IQR'] = np.round(val[1],2)
#         result_DF.loc[key, 'R2'] = np.round(val[2],2)

#         total.append(val[0])
#     total_MAE = np.round(np.mean(total),2) # SHOULD CORRECT THIS


#     total=[]
#     for key, val in singleRun.DF_err.items():
#         # print(np.round(np.mean(val['val_error']),2))
#         total.append(np.mean(val['val_error']))
#     total_bias = np.round(np.mean(total),2)

#     DF_err = pd.concat(singleRun.DF_err.values())
#     DF_err['val_error_sq'] = DF_err['val_error']**2
#     mse = np.sum(DF_err['val_error_sq']) / len(DF_err.val_error)
#     rmse = np.sqrt(mse)
#     # rmse

#     if print_results:
#             print(result_DF)
#             print('Average MAE over classes: ' + str(total_MAE) + ' uatm \n')

#     return [result_DF, total_MAE, total_bias, rmse]


# def summarize_class_errors(singleRun):


# %% Ancillary functions  

# def get_80_bounds(data):
#     mean = np.mean(data); std_dev = np.std(data)
#     low = mean - 1.282 * std_dev
#     high = mean + 1.282 * std_dev
#     print('80% bounds: \t' + str(low.round(5)) + ' to ' + str(high.round(5)) )
#     return [low, high]


# def get_95_bounds(data):
#     mean = np.mean(data); std_dev = np.std(data)
#     low = mean - 2 * std_dev
#     high = mean + 2 * std_dev
#     print('95% bounds: \t' + str(low.round(5)) + ' to ' + str(high.round(5)) )
#     return [low, high]

def get_depth_bias(data, ranges, var='val_error'):
    """ Get validation errors in 100m depth bins. """
    # Example range to pass as @param: ranges
    # pressure_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    #                 (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
    return {f"{start}-{end}": data[(data["pressure"] >= start) & (data["pressure"] < end)][var].values
            for start, end in ranges}

def print_errors_restrictdepth(data, var ='test_relative_error', pres_lim= [0,1000]):
    """ 
    @ param: data:  dataset that has "test_relative_error" in it
    """
    data = data[(data.pressure > pres_lim[0]) & (data.pressure < pres_lim[1])]
    err = data[var]
    print('Error metric: ' + var)
    print('Restricted to depths ' + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
    print('median abs error: \t' + str(np.abs(err).median()))
    print('mean abs error \t\t' + str(np.abs(err).mean()))

    # Bounds 95
    [low, high] = get_95_bounds(err)
    print('\n95% of errors fall between:')
    print(str(low.round(5)) + ' to ' + str(high.round(5)) )

    err = data[data.yearday <210][var]
    [low, high] = get_95_bounds(err)
    print("\nDuring SOGOS between depths " + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
    print('95% of errors fall between:')
    print(str(low.round(5)) + ' to ' + str(high.round(5)) )


# %% Calibration function s
# originally from 3.2_kfold
from scipy import stats

def apply_clustered_calibration(cal_coeffs, valDF, n_clusters, target_var = 'delta_pco2'):
    """ @param  data_byClass """
    
    # New version as of Mar 1 2026
    calibrated_valDF = valDF.copy()
    k_list = range(1, n_clusters+1)

    for ncluster in k_list:

        slope, intercept = cal_coeffs[ncluster]
        new_col = 'cluster' + str(ncluster) + '_calpred'
        calibrated_valDF[new_col] = calibrated_valDF['cluster' + str(ncluster) + '_pred'] * slope + intercept

    calibrated_valDF['lincal_pred'] = calibrated_valDF.apply(lambda row: weighted_prediction(row, k_list, pred_col_tag = '_calpred'),
                                                                      axis=1)

    calibrated_valDF['calibrated_error'] = calibrated_valDF['lincal_pred'] - calibrated_valDF[target_var]
    calibrated_valDF['calibrated_relative_error'] = calibrated_valDF['calibrated_error'] / calibrated_valDF[target_var]
       
    # === Version as of Feb 10 2026 (outdated)
    # Apply the calibration to the weighted prediction 
    # data_byClass = {ncluster:df for ncluster, df in valDF.groupby('cluster')}

    # for ncluster in range(1, n_clusters+1):
    #     class_valDF = data_byClass[ncluster].copy()
    #     slope, intercept = cal_coeffs[ncluster]

    #     # Apply the linear correction to the weighted_pred column
    #     class_valDF['lincal_pred'] = class_valDF['weighted_pred'] * slope + intercept

    #     class_valDF['calibrated_error'] = class_valDF['lincal_pred'] - class_valDF[target_var]
    #     class_valDF['calibrated_relative_error'] = class_valDF['calibrated_error'] / class_valDF[target_var]


    #     data_byClass[ncluster] = class_valDF.copy()
    
    # return data_byClass
    return calibrated_valDF


# # Make final predictions by weighting by class probabilities
# def weighted_prediction(row, ind_list):
#     """ ind_list or k_list is range(1,n_gmm+1)
#     """
#     total = 0
#     for k in ind_list:
#         prob_col = 'cluster' + str(k) + '_prob'
#         pred_col = 'cluster' + str(k) + '_pred'
#         total += row[prob_col] * row[pred_col]
#     return total



def get_clustered_calibration_coeffs(valDF):
    cal_coeffs = {ncluster:None for ncluster in valDF['cluster'].unique()}
    for ncluster, class_valDF in valDF.groupby('cluster'):
        class_valDF['n_decile'] = pd.qcut(class_valDF['weighted_pred'], 10, labels=list(range(1, 11))) #
        cal_pred = class_valDF.groupby('n_decile', observed=True)['weighted_pred'].agg(['mean', 'min', 'max', 'count'])
        cal_obs = class_valDF.groupby('n_decile', observed=True)['delta_pco2'].agg(['mean', 'min', 'max', 'count'])

        stat_var = 'mean'
        lincal = stats.linregress(cal_pred[stat_var].values, cal_obs[stat_var].values)

        cal_coeffs[ncluster] = [lincal.slope, lincal.intercept]
    
    return cal_coeffs

