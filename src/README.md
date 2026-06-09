# Code Directory for CRUSOE: 

## ./crusoe/

- Main modules to be packaged

## ./notebooks/

Preparing the data:

- [`./0.1_socat2024_processing.ipynb`](./0.1_socat2024_processing.ipynb): accessing SOCAT ship data
- [`./0.2_argopy_import.ipynb`](./0.2_argopy_import.ipynb): use Argopy package to download main training and validation data
- [`./0.3_coreArgo_processing.ipynb`](./0.3_coreArgo_processing.ipynb): 
- [`./0.4_bgcArgo_processing.ipynb`](./0.4_bgcArgo_processing.ipynb): 
- [`./0.6_add_MLD_ADT.ipynb`](./0.6_add_MLD_ADT.ipynb): add MLD (from Argo) and ADT to CoreArgo data

Preprocessing: (p1-processed)

- [`./1.2_P1_socat_features_with_colocation`](./1.2_P1_socat_features_with_colocation): Colocate SOCAT to nearest coreArgo float (after processing) and give MLD. Add ADT, solar.
Clustering (Gaussian mixture modeling):
- dev_float_processing (used to be 2.0)

- []()

- 


### Supporting Scripts

Major functions and custom classes are stored in modules. 

The `mod_RFR.py` holds the important class objects used in `Training_RandomForest.ipynb`. 


- *Modules*: 

                mod_RFR       as rfr            Main random forest regression methods
                mod_MLV       as mlv            Mixed layer variability and wavelet
                
                mod_main      as sg             Data and main ancillary functions.
                mod_L3proc    as gproc          Used for xarray Datset processing of the level 3, 'L3' gridded glider product
                mod_DFproc    as dfproc         Used for pandas Dataframe processing during analysis
                mod_plot      as sgplot         Used to define common plotting parameters; can reproduce all paper figs



Other scripts are used for troubleshooting or archiving additional information. \
(Generally not needed for non-author use.)

- *Supplementary*

        Supplement_RandomForest.ipynb   Additional RFR figures
                Supplement_Data_Log.ipynb       Notes on different data streams 



### Variable Naming Conventions

- *Glider:*    

                dav_659                         Profiles-averaged metrics, incl. MLD
                df_659                          Flattened dataframe
                profid                          Unique name for each profile

- *Floats:*    

                dav_659                         Profile-averaged metrics, incl. MLD
                df_659                          Flattened dataframe
                wmoid                           Unique name for each float WMO#
                profid                          Unique name for each profile


## Other Notes

Two phases of processing were developed externally in MATLAB: 

- Oxygen optode time response correction (Courtesy of Yui Takeshita, MBARI)
- ESPER-Mixed Prediction ([Carter et al. 2021](https://doi-org.offcampus.lib.washington.edu/10.1002/lom3.10461))
- CANYON-B Prediction ([Bittig et al. 2018](https://doi.org/10.3389/fmars.2018.00328))
- ACC Front locations courtesy of [Sauve et al. 2023](https://doi.org/10.1029/2023JC019815)

To process Argo data, we use code courtesy of the [GO-BGC Toolbox](https://github.com/go-bgc/workshop-python/blob/main/GO_BGC_Workshop_Python_tutorial.ipynb) (Ethan Campbell 2021).