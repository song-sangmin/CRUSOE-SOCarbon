import seaborn as sns 
import mod_ocean as myocn

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors     as mpcolors
import cartopy.crs as ccrs

import mod_plotting as mod_plot
import mod_regression as mod_reg
import figs_pcm


# datatype_colors = {'train' : "#006F83", 
#                    'val' :  "#FA4B4B", 
#                    'combined' : "#373232",
#                    'train_float': "#D98600",  # rose
#                    'val_float': "#920EC2",  # rose
#                    'train_ship': "#395AEC",  # dark blue
#                    'val_ship': "#2A13A0",  # dark blue
#                    }


datatype_colors = {'train' : "#332288", 
                   'val' :  "#88CCEE", 
                   'combined' : "#524F4F",
                   'train_comb':  "#332288", 
                   'val_comb':  "#88CCEE", 
                   'train_float': "#882255",  # rose
                   'val_float': "#DDCC77",  # rose
                   'train_ship': "#44AA99",  # dark blue
                   'val_ship': "#CC6677",  # dark blue
                   'trainval_float': "crimson",   # 
                   'trainval_ship':  "#2441AA",  #
                   'test' : "#076335" 
                   }


plat_colors = {'bgc': "crimson", 
               'socat' : "#96ADFF", 
               'core' : "#EBD367" 
                   }

def tol8():
    tol_palette = [
        "#332288",  # dark blue
        "#88CCEE",  # light blue
        "#44AA99",  # teal
        "#117733",  # green
        "#999933",  # olive
        "#DDCC77",  # sand
        "#CC6677",  # rose
        "#882255"   # wine
    ]


res_delta_tag = 'Residual Δ-pCO$_{2}$ (µatm)'

# %% TRACER DISTRIBUTIONS

def overlay_histograms_singleVar(plotdict, plotvar, ax=None,
                                 nbins=50, binwidth=10, alpha=0.8, stat_type = 'count', showLegend=True):
    
    # binwidth will override nbins
    if ax is None:
        fig = plt.figure(figsize=(8,5)); ax = fig.gca()

    for tag, platdf in plotdict.items():
        # try: ax.hist(platdf[plotvar], bins=nbins, density=True, alpha=alpha, label=tag, color=datatype_colors[tag], zorder=5);
        # except: ax.hist(platdf[plotvar], bins=nbins, density=True, alpha=alpha, label=tag, zorder=5)
        # alpha = alpha - 0.1
    
    
        if binwidth is None:
            sns.histplot(platdf[plotvar], bins=nbins, kde=True, stat=stat_type, ax=ax, color=datatype_colors[tag], alpha=0.3)
        else:
            sns.histplot(platdf[plotvar], binwidth=binwidth, kde=True, stat=stat_type, ax=ax, color=datatype_colors[tag], alpha=0.4)
        # ax.hist(plot_data['val_error'], bins=nbins)


    ax.grid(linestyle='--', alpha=0.5, zorder=0)
    if showLegend: ax.legend()

    return ax


# %% SPATIAL DISTRIBUTIONS / MAPS

# %% TEMPORAL DISTRIBUTIONS IN DATASET

# Probabilities, Panel across classes
# nclasses = 8

# fig, axs = plt.subplots(2,4, figsize=(15,6), layout='tight')
# # for ax in axs.flatten():

# for ind, ax in enumerate(axs.flatten()):
#     # temp = training_classes[ind]
#     foo = myocn.expand_datetime(training_classes[ind+1], type='dataframe')
#     sns.countplot(x='month', data=foo, ax=ax)
#     ax.set_title('Class ' + str(ind+1), fontsize=14)


def countplot_months_byClass():
    ind_list = [x for x in storedRuns_gmm8_ex3[run_tag].ind_list]
    nclasses = len(ind_list)

    fig, axs = plt.subplots(2,4, figsize=(15,6), layout='tight')
    axs = axs.flatten()

    for key, val in enumerate(ind_list):
        ax = axs[key]

        # # training
        # foo = myocn.expand_datetime(trainClasses[val], type='dataframe')
        # sns.countplot(x='month', data=foo, ax=ax, color = 'lightblue')
        
        # validation
        foo = myocn.expand_datetime(valClasses[val], type='dataframe')
        sns.countplot(x='month', data=foo, ax=ax, color = 'crimson')

        ax.set_title('Class ' + str(val), fontsize=14)


# Colored points, by month 
def maps_paneled_byYear():
    fig, axs = plt.subplots(2,5, figsize=(17, 10), layout='tight', subplot_kw={'projection': ccrs.SouthPolarStereo()})

    for ax in axs.flatten():
        myplt.setup_SO_axes(ax, fig , 
                                add_gridlines         = False, 
                                color_land            = True,
                                land_facecolor        = land_facecolor,
                                land_edgecolor        = land_edgecolor,
                                fontsize              = 14,
                                max_latitude         = -35,
                                map_facecolor         = 'white',
                                coast_linewidth       = coast_linewidth)

        ### Add front and sea ice edge
        # stf_patch  = plt.Polygon(stf,  fill=False, edgecolor='grey',   zorder=15)
        # saf_patch  = plt.Polygon(saf,  fill=False, edgecolor='grey',   zorder=14)
        # pf_patch   = plt.Polygon(pf,   fill=False, edgecolor='grey',    zorder=13)
        # sacc_patch = plt.Polygon(sacc, fill=False, edgecolor='grey',  zorder=12)
        # sie_patch  = plt.Polygon(sie,  fill=True,  edgecolor='grey',   zorder=0,  facecolor='darkgrey', alpha=0.4)

        # ax.add_patch(stf_patch)
        # ax.add_patch(saf_patch)
        # ax.add_patch(pf_patch)
        # ax.add_patch(sacc_patch)
        # ax.add_patch(sie_patch)

    for ind, key in enumerate(np.sort(list(annual_profiles.keys()))):
        # colormesh histogram
        argo_ave = annual_profiles[key]

        bin_values,_,__ = np.histogram2d(argo_ave.LONGITUDE.values,argo_ave.LATITUDE.values,bins=(lon_bins, lat_bins) )
        X, Y = np.meshgrid(lon_bins, lat_bins)

        # ax1.scatter(idf.longitude, idf.latitude, s=0.5, transform=ccrs.PlateCarree(), label=key)
        if ind <5:
            axs[0,ind].scatter(argo_ave.LONGITUDE.values,argo_ave.LATITUDE.values, s=0.1, alpha=0.35, transform=ccrs.PlateCarree(), label=key)
            # hist2d(argo_ave.LONGITUDE.values,argo_ave.LATITUDE.values,bins=(lon_bins, lat_bins), vmax=vmax, cmap=cmo.amp,
            #                   transform = ccrs.PlateCarree())
            axs[0,ind].set_title(key) #+ ' Core Argo profile counts'
            print(key)
        else:
            axs[1,ind-5].scatter(argo_ave.LONGITUDE.values,argo_ave.LATITUDE.values, s=0.1, alpha=0.35, transform=ccrs.PlateCarree(), label=key)
            # axs[1, ind-5].hist2d(argo_ave.LONGITUDE.values,argo_ave.LATITUDE.values,bins=(lon_bins, lat_bins), vmax=vmax, cmap=cmo.amp, 
            #                      transform = ccrs.PlateCarree())
            axs[1, ind-5].set_title(key)

