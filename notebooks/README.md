
# Notebook Directory 


## [`./notebooks/`](./notebooks/)

Download data, quality control, reformat:
(Outputs are L3-masked files; masked to 1000m bathymetry)
- [`0.1_socat2024_processing.ipynb`](./notebooks/0.1_socat2024_processing.ipynb): accessing SOCAT ship data
- [`0.2_argopy_import.ipynb`](./notebooks/0.2_argopy_import.ipynb): use Argopy package to download main training and validation data
- [`0.3_coreArgo_processing.ipynb`](./notebooks/0.3_coreArgo_processing.ipynb):
- [`0.4_bgcArgo_processing.ipynb`](./notebooks/0.4_bgcArgo_processing.ipynb):

Preprocessing and adding features (mixed layer depth, sea surface height, solar radiation): 
(Outputs are P1-processed files)
- [`1.1_P1_coreArgo_feature_processing.ipynb`](./notebooks/1.1_P1_coreArgo_feature_processing): add MLD and ADT to coreArgo data (long run time)
- [`1.2_P1_bgcArgo_feature_processing.ipynb`](./notebooks/1.2_P1_bgcArgo_feature_processing): for bgcArgo data
- [`1.3_P1_socat_feature_processing.ipynb`](./notebooks/1.2_P1_bgcArgo_feature_processing): add MLD (from Argo) and ADT to socat data

Unsupervised clustering (principal component analysis + gaussian mixture modeling) 
(Outputs are P2-clustered files)
- [`2.1_PCM_fit_coreArgo_2014–2024.ipynb`](./notebooks/2.1_PCM_fit_coreArgo_2014–2024): classify coreArgo data, including test data from 2024
- [`2.2_PCM_classify_P1processed_bgcArgo_socat.ipynb`](./notebooks/2.2_PCM_classify_P1processed_bgcArgo_socat.ipynb): use classifications from 2.1_PCM to sort bgcArgo and SOCAT observations

Cluster-dependent, supervised regression (random forest, xgboost, extremely randomized trees)
- [`3.1_simple_holdout_clustered_RFR.ipynb`](./notebooks/3.1_simple_holdout_clustered_RFR.ipynb): Fastest training for preliminary comparisons. validated on simple 20% withheld split of data (split by platform)
- [`3.2_kfold_clustered_RFR.ipynb`](./notebooks/3.2_kfold_clustered_RFR.ipynb): clustered k-fold validation, all samples represented in validation once by iteratively withholding folds. Slower than 3.1, still fast enough to run locally.
- [`3.3_MAIN_regression_nested_crossval_tuning.ipynb`](./notebooks/3.3_MAIN_regression_nested_crossval_tuning.ipynb) comprehensive cross-validation with nested parameter tuning (outer and inner folds); implements all three algorithms (RFR, XGB, ERT). Slow to run (parallelization in progress).


## Naming Conventions

- *Platform data:*    

                coreDATA                         Subsurface profile data
                coreINDEX                        Surface only data (useful for mapping, indexing)
                profid                           Unique name for each profile
                wmoid                            Argo float WMO ID (many profids per 1 wmoid)
                cruiseid                         Unique name for each SOCAT cruise or continuous series

- *Measured variables:*    

                pK1_pHbias5                     Corrected pK1 (Johnson et al., in prep) and 5mpH correction
                solar_rad                       Maximum daily solar radiation, computed from solarpy
                wind_speed                      ERA5 computed value from hourly data, sqrt(u2+v2)
                atmos_pres_atm                  atmospheric sea level pressure, in atmospheres

## Cross-validation strategy (see 3.3)

To evaluate one algorithm/feature list (performance estimation level)
* For each Southern Ocean GMM class, create object with X outer folds + its own inner nested CV (```soclass_cvtainers```)
* For each outer fold: 
    * Withhold one fold (`fold_validationDF`), use rest of the outer fold data (`fold_trainDF`) for the inner loop, which then considers new inner folds for hyperparameter tuning
    * Find best hyperparameters by using GridSearchCV on inner folds (new partitions from `fold_trainDF`)
    * Each fold will be assigned a “best estimator” that is refit on all the given tuning data by default (`fold_trainDF`); store best parameters

Once all the classes’ models have been trained/tuned: 
To get weighted validation error (weighted avg. over the clusters), each fold is considered separately (`fold_validationDF`)
* Fold 1: Get class1, fold1 estimator and make prediction. 
* Repeat for class2, fold1 estimator. 
* Take weighted average; difference between weighted average and true value is our estimate of error
* Repeat for Folds 2-5 and combine

For each run tag, we have five fold MAEs (`RFR_OuterFoldResults`) and a combined weighted validation dataset (represents the entirety of data `RFR_CVResults`) 
