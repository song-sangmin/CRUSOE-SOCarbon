import mod_ocean as myocn
import mod_plotting as myplt
import mod_pca 
import mod_loading as loader

from   sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM

import pandas as pd

def main():
    # Load data
    coreDS, coreINDEX = loader.import_data(type=['core'])
    centered_coreDS = mod_pca.center_byPressure(coreDS, with_scaling = False)

    # === Create PCA input dataframe, choose pressure levels to include
    n_pca = 6
    ncomps = 8
    dbar_limit = 1281
    [Xtrain, Xtrans, pca_obj, PCdict, gmm, Y_gmm, class_locs, class_data] = mod_pca.run_pca_and_gmm(
                                                                    centered_coreDS, coreDS, coreINDEX,
                                                                    n_pca=n_pca, 
                                                                    n_gmm=ncomps, 
                                                                    dbar_limit=dbar_limit
                                                                )
    
    post_probs, class_probs = mod_pca.get_posterior_probs(gmm,Xtrans, Xtrain, Y_gmm, class_locs)


if __name__ == "__main__":
    main()



# def main():
#     print("Starting analysis...")
#     ds = module_utils.load_data("data/raw/")
#     result = module_regression.fit_model(ds)
#     module_plotting.save_figures(result)
#     module_utils.save_output(result, "output/results/")

# if __name__ == "__main__":
#     main()