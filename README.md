# CRUSOE: Southern Ocean Carbon Reconstruction

Contact: Song Sangmin <sangsong@uw.edu> \
University of Washington, School of Oceanography

Last updated: Jun 8 2026

## Overview


## Usage notes

To run locally, install the environment `crusoe-dev` using the file `environment.yml`.


## [Notebooks](./src/notebooks/)

Analyses are in Jupyter notebook, see notebook directory. 
The main file for model development is:

- [`3.3_MAIN_regression_nested_crossval_tuning.ipynb`](./src/notebooks/3.3_MAIN_regression_nested_crossval_tuning.ipynb) comprehensive cross-validation with nested parameter tuning (outer and inner folds); implements all three algorithms (RFR, XGB, ERT). 


<!-- ### Folder Directory

- `scripts/` : code for analysis
- `data/` : float, ship, and glider data as downloaded
- `working-vars/` : calculated output variables from analysis
- `images/` : final output figures
 -->

## Data Sources

- Argo data were accessed with [Argopy](https://argopy.readthedocs.io/en/latest/) from the Jan 2025 GDAC
- Satellite data from Copernicus Marine: (https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/download) (cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_adt-sla)
- Wind speed from Copernicus Marine: [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview](ERA5)
- Bathymetry data from ETOPO2
- Solar radiation from Solarpy
- Atmospheric xCO2 from 
- Sea ice concentration from NSIDC
- Sea level pressure from GLOBALVIEW obspack Marine Boundary Layer product

## Funding 

This work was funded by SOCCOM3.