# CRUSOE: Southern Ocean Carbon Reconstruction

Contact: Song Sangmin <sangsong@uw.edu> \
University of Washington, School of Oceanography

Last updated: Jun 8 2026

## Overview


## Usage notes

To run locally, install the environment `crusoe-dev` using the file `environment.yml`.


## [Notebooks](./src/notebooks/)

Download data, quality control, reformat:
(Outputs are L3-masked files; masked to 1000m bathymetry)
- [`0.1_socat2024_processing.ipynb`](./src/notebooks/0.1_socat2024_processing.ipynb): accessing SOCAT ship data
- [`0.2_argopy_import.ipynb`](./src/notebooks/0.2_argopy_import.ipynb): use Argopy package to download main training and validation data
- [`0.3_coreArgo_processing.ipynb`](./src/notebooks/0.3_coreArgo_processing.ipynb):
- [`0.4_bgcArgo_processing.ipynb`](./src/notebooks/0.4_bgcArgo_processing.ipynb):

Preprocessing and adding features (mixed layer depth, sea surface height, solar radiation): 
(Outputs are P1-processed files)
- [`1.1_P1_coreArgo_feature_processing.ipynb`](./src/notebooks/1.1_P1_coreArgo_feature_processing): add MLD and ADT to coreArgo data (long run time)
- [`1.2_P1_bgcArgo_feature_processing.ipynb`](./src/notebooks/1.2_P1_bgcArgo_feature_processing): for bgcArgo data
- [`1.3_P1_socat_feature_processing.ipynb`](./src/notebooks/1.2_P1_bgcArgo_feature_processing): add MLD (from Argo) and ADT to socat data

Unsupervised clustering (principal component analysis + gaussian mixture modeling) 
(Outputs are P2-clustered files)
- [`2.1_PCM_fit_coreArgo_2014–2024.ipynb`](./src/notebooks/2.1_PCM_fit_coreArgo_2014–2024): classify coreArgo data, including test data from 2024
- [`2.2_PCM_classify_P1processed_bgcArgo_socat.ipynb`](./src/notebooks/2.2_PCM_classify_P1processed_bgcArgo_socat.ipynb): use classifications from 2.1_PCM to sort bgcArgo and SOCAT observations

Cluster-dependent, supervised regression (random forest, xgboost, extremely randomized trees)
- [`3.1_simple_holdout_clustered_RFR.ipynb`](./src/notebooks/3.1_simple_holdout_clustered_RFR.ipynb): Fastest training for preliminary comparisons. validated on simple 20% withheld split of data (split by platform)
- [`3.2_kfold_clustered_RFR.ipynb`](./src/notebooks/3.2_kfold_clustered_RFR.ipynb): clustered k-fold validation, all samples represented in validation once by iteratively withholding folds. Slower than 3.1, still fast enough to run locally.
- [`3.3_MAIN_regression_nested_crossval_tuning.ipynb`](./src/notebooks/3.3_MAIN_regression_nested_crossval_tuning.ipynb) comprehensive cross-validation with nested parameter tuning (outer and inner folds); implements all three algorithms (RFR, XGB, ERT). Slow to run (parallelization in progress).


<!-- ### Folder Directory

- `scripts/` : code for analysis
- `data/` : float, ship, and glider data as downloaded
- `working-vars/` : calculated output variables from analysis
- `images/` : final output figures
 -->

### Data Sources

- Argo data were accessed with [Argopy](https://argopy.readthedocs.io/en/latest/) from the Jan 2025 GDAC
- Satellite data from Copernicus Marine: (https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/download) (cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_adt-sla)
- Wind speed from Copernicus Marine: [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview](ERA5)
- Bathymetry data from ETOPO2
- Solar radiation from Solarpy
- Atmospheric xCO2 from 
- Sea ice concentration from NSIDC
- Sea level pressure from GLOBALVIEW obspack Marine Boundary Layer product
