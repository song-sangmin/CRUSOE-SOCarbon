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

# %% Initialize variables and data

def train_RFR(feat_list, 
              training, validation,
              var_predict = 'delta_fco2',
              cols_ignore_nans = [],
              ntrees=1000, max_feats = 1/3, min_samples_split=5):
    """ 
    Main method to train the RF model.
    Update 04.18.24: Redo bootstrapping. 
    @param: 
        feat_list: list of features to use in the model
        training: training data unscaled, i.e. original range of values
        validation: validation data unscaled
        test: test data unscaled 
        ntrees: 1000 trees by default.

    @return:
        Mdl: trained RF model
        Mdl_results: 
        Mdl_prediction: Dataframe with error metrics for the *VALIDATION* set
    """

    Mdl = RandomForestRegressor(ntrees, max_features = max_feats, random_state = 0, bootstrap=True, min_samples_split=min_samples_split)
        #  max_features: use at most X features at each split (m~sqrt(total features))

    # Drop NaN's without profid or wmoid
    # cols_na = [col for col in training.columns if col not in cols_ignore_nans] 
    training_nona = training #.dropna(axis=0, subset=cols_na)  
    validation_nona = validation #.dropna(axis=0, subset=cols_na)

    # Train the model 
    X_training = training_nona[feat_list].to_numpy()
    Y_training = training_nona[var_predict].to_numpy()
    Mdl.fit(X_training, Y_training)

    # Estimate and get error predicts
    Y_pred_validation = Mdl.predict(validation_nona[feat_list].to_numpy())
    AE_RF_validation = np.abs(Y_pred_validation - validation_nona[var_predict])
    IQR_RF_validation = iqr(abs(AE_RF_validation))
    r2_RF_validation = r2_score(validation_nona[var_predict], Y_pred_validation)

    # Validation error DF to return
    Mdl_prediction = validation_nona.copy() #pd.DataFrame()
    Mdl_prediction['val_prediction'] = Y_pred_validation
    Mdl_prediction['val_error'] = Y_pred_validation - validation_nona[var_predict]
    Mdl_prediction['val_relative_error'] = Mdl_prediction['val_error']/validation_nona[var_predict]

    Mdl_results = [np.nanmedian(abs(AE_RF_validation)), IQR_RF_validation, r2_RF_validation]

    return [Mdl, Mdl_results, Mdl_prediction]


def test_RFR(feat_list, Mdl, 
              test, 
              var_predict = 'delta_fco2'):  
    """ For running model once and testing on multiple synthetic test sets."""
    # cols_na = [col for col in training.columns if col not in ['profid', 'wmoid', 'AOU', 'dist_maxb']]
    # test_nona = test.dropna(axis=0, subset=cols_na)
    test_nona = test.dropna(axis=0, subset=feat_list)
    X_test = test_nona[feat_list].to_numpy()
    Y_test = test_nona[var_predict].to_numpy()
    Y_pred_test = Mdl.predict(X_test)

    # Create dataframe for the test set with depth --> 
    Mdl_test_error = test_nona.copy(); 
    Mdl_test_error = Mdl_test_error.reset_index(drop=True)
    observed = Mdl_test_error[var_predict].to_numpy()

    # Save new dataframe with test results
    Mdl_test_error['test_prediction'] = Y_pred_test
    Mdl_test_error['test_error'] = Mdl_test_error['test_prediction'] - observed
    Mdl_test_error['test_relative_error'] = Mdl_test_error['test_error']/observed

    # AE_RF_test = np.abs(Y_pred_test - test_nona[var_predict]) # abs val of Mdl_test_error['test_error']
    # IQR_RF_test = iqr(abs(AE_RF_test))
    # r2_RF_test = r2_score(test_nona[var_predict], Y_pred_test)

    return Mdl_test_error



# %% NEW METHODS FOR CLASS-DEPENDENT REGRESSION
# UPDATED DEC 2025

def fit_cluster_RFR(feat_list, 
              trainingDF, 
              var_predict = 'delta_fco2',
              ntrees=1000, max_feats = 1/3, min_samples_split=5):
    """ 
    Fit single RFR for a class. 
    Compute validation errors later 
    """
    Mdl = RandomForestRegressor(ntrees, max_features = max_feats, random_state = 0, bootstrap=True, min_samples_split=min_samples_split)
        #  max_features: use at most X features at each split (m~sqrt(total features))

    # Train the model 
    X_training = trainingDF[feat_list].to_numpy()
    Y_training = trainingDF[var_predict].to_numpy()
    Mdl.fit(X_training, Y_training)

    return Mdl

# def fit_cluster_XGB(feat_list, 
#               trainingDF, 
#               var_predict = 'delta_fco2',
#               ntrees=1000, max_feats = 1/3, min_samples_split=5):
#     """ 
#     Fit single RFR for a class. 
#     Compute validation errors later 
#     """
#     Mdl = XGBRegressor()

#     # Train the model 
#     X_training = trainingDF[feat_list].to_numpy()
#     Y_training = trainingDF[var_predict].to_numpy()
#     Mdl.fit(X_training, Y_training)

#     return Mdl




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


def subset_training_validation(platDF, indexer = 'profid', split_frac = 0.8):
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




# %% Classes for storing model results

class ModelVersion: 
    """ 
    Object holding different RF model runs. 
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
    Object holding models for each separate class
    ind_list = classes (i.e. range(1,9))
    """
    def __init__(self, ind_list):
        self.ind_list = ind_list
        self.models = {model: None for model in ind_list}
        self.errorDF = None # validation DF with errors
        self.model_metrics = None

def get_ModelVersion_results(RF_run, print_results=True):
    """ RF_run is one ModelVersion object
    RF_run = storedRuns[run_tag]"""
    result_DF = pd.DataFrame(index=range(1,9), columns=['MAE', 'IQR', 'R2', 'mean_bias'])
    total = []
    for key, val in RF_run.errorstats.items():
        # print('Class ' + str(key) + ': MAE=' + str(np.round(val[0],2)) + ', IQR=' + str(np.round(val[1],2)) + ', R2=' + str(np.round(val[2],2)) + '')
        result_DF.loc[key, 'MAE'] = np.round(val[0],2)
        result_DF.loc[key, 'IQR'] = np.round(val[1],2)
        result_DF.loc[key, 'R2'] = np.round(val[2],2)

        total.append(val[0])
    total_MAE = np.round(np.mean(total),2)

    total=[]
    for key, val in RF_run.DF_err.items():
        # print(np.round(np.mean(val['val_error']),2))
        total.append(np.mean(val['val_error']))
    total_bias = np.round(np.mean(total),2)

    DF_err = pd.concat(RF_run.DF_err.values())
    DF_err['val_error_sq'] = DF_err['val_error']**2
    mse = np.sum(DF_err['val_error_sq']) / len(DF_err.val_error)
    rmse = np.sqrt(mse)
    # rmse

    if print_results:
            print(result_DF)
            print('Average MAE over classes: ' + str(total_MAE) + ' uatm \n')


    return [result_DF, total_MAE, total_bias, rmse]


    # def get_results(self):
    #     model_metrics = pd.DataFrame()
    #     model_metrics['ind'] = self.ind_list
    #     model_metrics['MAE'] = [x[2] for x in self.errorstats.values()]
    #     return model_metrics
    
    # def get_errors(self):
    #     model_metrics = pd.DataFrame()
    #     model_metrics['ind'] = self.ind_list
    #     model_metrics['MAE'] = [x[2] for x in self.errorstats.values()]
    #     return model_metrics
    
    # def get_metrics(self):
    #     model_metrics = pd.DataFrame()
    #     model_metrics['ind'] = self.ind_list
    #     model_metrics['validation_MAE'] = [x[1][1].item() for x in self.MAE.items()]
    #     model_metrics['validation_IQR'] = [x[1][1].item() for x in self.IQR.items()]
    #     model_metrics['validation_r2'] = [x[1][1].item() for x in self.r2.items()]
    #     model_metrics['test_MAE'] = [x[1][2].item() for x in self.MAE.items()]
    #     model_metrics['test_IQR'] = [x[1][2].item() for x in self.IQR.items()]
    #     model_metrics['test_r2'] = [x[1][2].item() for x in self.r2.items()]
    #     model_metrics.set_index('ind', inplace=True)
        
    #     return model_metrics

# %% Classes for storing cross-validation results
    
# class KFold:
#     """
#     From a single KFold run (using one model list)
#     Object holds the data from all folds
#     (indexed by fold number, e.g. 1, 2, 3...)
#     """
#     def __init__(self, nfolds=10):
#         fold_list = np.arange(1,nfolds+1)
#         self.fold_list = fold_list # model_list, 
#         self.folds = {k: None for k in fold_list} # models
#         self.MAE = {k: None for k in fold_list}
#         self.IQR = {k: None for k in fold_list}
#         self.r2 = {k: None for k in fold_list}
#         self.DF_err = {k: None for k in fold_list}
#         self.val_err = {k: None for k in fold_list}
    
#     def get_metrics(self):
#         folds_metrics = pd.DataFrame()
#         folds_metrics['fold'] = self.fold_list
#         folds_metrics['validation_MAE'] = [x[1][1].item() for x in self.MAE.items()]
#         folds_metrics['validation_IQR'] = [x[1][1].item() for x in self.IQR.items()]
#         folds_metrics['validation_r2'] = [x[1][1].item() for x in self.r2.items()]
#         folds_metrics['test_MAE'] = [x[1][2].item() for x in self.MAE.items()]
#         folds_metrics['test_IQR'] = [x[1][2].item() for x in self.IQR.items()]
#         folds_metrics['test_r2'] = [x[1][2].item() for x in self.r2.items()]
#         folds_metrics.set_index('fold', inplace=True)
        
#         return folds_metrics

class CrossVal_KFold:
    """ 
     Larger object containing CV information across all models in model_list
     (indexed by model, e.g. 'Model_X')
     During k-fold, we combine errors across folds for each model, 
     such that we add 10% of validation errors from each fold,
     (represent 100% of training data)
    """
    def __init__(self, model_list):
        self.model_list = model_list
        self.val_DF = {k: None for k in model_list} # full pd DF

        # Series of all errors, for plotting histograms
        self.val_error = {k: None for k in model_list} # series of all errors
        self.val_relative_error = {k: None for k in model_list}
        # List of k MAEs. Mean/STD are for Table 1: Val Errors
        self.MAEs = {k: None for k in model_list}
        self.IQRs = {k: None for k in model_list}

    def get_metrics(self, model_list=None): 
        """ 
        Calculate metrics.
        Notice different metrics from the ModelVersion class.
        Here, we want to combine data across the 10 folds for each model,
        i.e. aggregate errors across folds, indexed by Model_X"""
        cvk_metrics = pd.DataFrame()
        # Fill in metrics
        if model_list == None:
            model_list = self.model_list
        
        for ind, mdl in enumerate(model_list):
            cvk_metrics.at[ind, 'median_MAEs'] = np.nanmedian(self.MAEs[mdl].values) # median of 10 medians from folds
            cvk_metrics.at[ind, 'mean_MAEs'] = np.nanmean(self.MAEs[mdl].values)
            cvk_metrics.at[ind, 'mean_IQRs'] = iqr(self.IQRs[mdl].values)

            ab_err = np.abs(self.val_error[mdl].values)
            cvk_metrics.at[ind, 'total_median_AE'] = np.nanmedian(ab_err) # median of all combined errors across folds
            cvk_metrics.at[ind, 'total_mean_AE'] = np.nanmean(ab_err)
            cvk_metrics.at[ind, 'total_IQR'] = iqr(ab_err)
            cvk_metrics.at[ind, 'total_median_bias'] = np.nanmedian(self.val_error[mdl].values)

        cvk_metrics['model'] = self.model_list
        cvk_metrics = cvk_metrics.set_index('model') # .fillna(np.nan)

        return cvk_metrics

def split_kfolds(platDF, nfolds = 10):
        """
        @param platDF: scaled dataframe with all float and ship profiles
                nfolds: number of folds to split data into. 
                        Each fold will be used for validation once.
        @return folds_training: list of kfold dataframes for training
                folds_validation: list of kfold dataframes for validation

        """
        profs = pd.unique(platDF.profid)

        if 5906030 in profs:
                print('warning: 5906030 is in platDF')

        # Shuffle float profile ID's
        # Each element of fold_profs is a list of profile ID's belonging to the k-th fold
        np.random.seed(42) 
        rng = np.random.default_rng(); rng.shuffle(profs)
        fold_profs = np.array_split(profs, nfolds)

        # Make a dictionary of dataframes for validation and training. 
        # Each fold (1/nth of the total training data) will be used for validation once.
        # All profiles that are not part of that fold are used for training. 
        training_list = []; validation_list = []
        for i in np.arange(0,nfolds):
                df = platDF[platDF['profid'].isin(fold_profs[i])].copy()
                validation_list.append(df)

                df = platDF[~platDF['profid'].isin(fold_profs[i])].copy()
                training_list.append(df)

                
        folds_validation = {k:v for k, v in zip(np.arange(1,nfolds+1), validation_list)}
        folds_training = {k:v for k, v in zip(np.arange(1,nfolds+1), training_list)}

        return folds_training, folds_validation

def split_loos(wmos, floatDF, shipDF):
    loo_training = dict.fromkeys(wmos)
    loo_validation = dict.fromkeys(wmos)

    for withheld in wmos:
        floatdat = floatDF[(floatDF.wmoid!=withheld) & (floatDF.wmoid!=5906030)]
        loo_training[withheld] = pd.concat([floatdat, shipDF], ignore_index=True)
        loo_validation[withheld] = floatDF[floatDF.wmoid==withheld]

    return loo_training, loo_validation


# %% Ancillary functions  

def get_80_bounds(data):
    mean = np.mean(data); std_dev = np.std(data)
    low = mean - 1.282 * std_dev
    high = mean + 1.282 * std_dev

    return [low, high]


def get_95_bounds(data):
    mean = np.mean(data); std_dev = np.std(data)
    low = mean - 2 * std_dev
    high = mean + 2 * std_dev

    return [low, high]

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


def print_errors(data, var ='test_relative_error'):
    """ 
    @ param: data:  dataset that has "test_relative_error" in it
    """
    err = data[var]
    print('Error metric: ' + var)
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

