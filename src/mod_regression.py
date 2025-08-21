import pandas as pd
import xarray as xr
import numpy as np
import scipy
import math
from scipy.stats import iqr


# %% Initialize variables and data

# %% Methods for preparing ML Data

def split_training_data(platDS, indexer = 'profid', test_split = False):
    """ 
    @param platDS 
        indexer: name of dimension to use for splitting ('profid' for Argo, 'sid' for SOCAT)
    @return: 
        training_data: training data (80% of profiles) 
        validation_data: validation data (20% of profiles)
    
    This method treats a float profile like the smallest discrete "unit" upon which to train and validate the model.
    Our goal is to predict a water column (profile) rather than a discrete observation, so profiles are kept together while splitting.
    Note that test data will be all profiles from the SOGOS float when test_split is set as False by default.
    """
    # Create list of unique profile ID's and select random 80% for training
    ids = pd.unique(platDS[indexer].values)

    np.random.seed(42) 
    training_ids = np.random.choice(ids, int(np.floor(0.8*len(ids))), replace=False)
    training_data = platDS.where(platDS[indexer].isin(training_ids), drop=True)

    if test_split == False:
        # Take remaining profiles that were not in training set, for validation
        validation_ids = [x for x in ids if x not in training_ids]
        validation_data = platDS.where(platDS[indexer].isin(validation_ids), drop=True)
        test_data = [] # will become SOGOS data

    else:
        # Take HALF of remaining profiles that were not in training set, for validation (10% of total)
        ids_vt = [x for x in ids if x not in training_ids]  # remaining profiles after training data removed
        validation_ids= np.random.choice(ids_vt, int(np.floor(0.5*len(ids_vt))), replace=False)
        validation_data = platDS.where(platDS[indexer].isin(validation_ids), drop=True)

        test_ids = [x for x in ids_vt if x not in  validation_ids]
        test_data = platDS.where(platDS[indexer].isin(test_ids), drop=True)

    return training_data, validation_data, test_data
# %% Classes for storing model results

class ModelVersion: 
    """ 
    Object holding different RF model runs. 
    """
    def __init__(self, ind_list):
        self.ind_list = ind_list
        self.models = {model: None for model in ind_list}
        self.MAE = {model: None for model in ind_list}
        self.IQR = {model: None for model in ind_list}
        self.r2 = {model: None for model in ind_list}
        self.DF_err = {model: None for model in ind_list}
        self.val_err = {model: None for model in ind_list}
    
    def add(self, model_name):
        self.models[model_name] = None
        self.MAE[model_name] = None
        self.IQR[model_name] = None
        self.r2[model_name] = None
        self.DF_err[model_name] = None
        self.val_err[model_name] = None
    
    def copy(self):
        return self
    
    def get_metrics(self):
        model_metrics = pd.DataFrame()
        model_metrics['ind'] = self.ind_list
        model_metrics['validation_MAE'] = [x[1][1].item() for x in self.MAE.items()]
        model_metrics['validation_IQR'] = [x[1][1].item() for x in self.IQR.items()]
        model_metrics['validation_r2'] = [x[1][1].item() for x in self.r2.items()]
        model_metrics['test_MAE'] = [x[1][2].item() for x in self.MAE.items()]
        model_metrics['test_IQR'] = [x[1][2].item() for x in self.IQR.items()]
        model_metrics['test_r2'] = [x[1][2].item() for x in self.r2.items()]
        model_metrics.set_index('ind', inplace=True)
        
        return model_metrics

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

def print_errors(data, var ='test_relative_error', pres_lim= [0,1000]):
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



