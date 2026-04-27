# Main use in 1.1_pca_transform.ipynb

import matplotlib
import matplotlib.pyplot     as plt  
import mod_plotting as myplt
from   cmocean               import cm as cmo
import cartopy.crs           as     ccrs    
from importlib import reload

from mod_ocean import expand_datetime
import mod_southpolarplot as sopo

# tol_palette = myplt.tol8()
# gmm_palette = [
#     "#332288",  # dark blue   
#     "#CC6677",  # rose
#     "#DDCC77",  # sand
#     "#2E8783",  # teal
#     "#94235C",   # wine
#     "#999933",  # olive
#     "#83DAFF", # light blue
#     "#11703E"  # green
# ]

# light version for black background
gmm_palette = [
    "#83DAFF",  # dark blue   
    "#DDCC77",  # sand
    "#E47C8D",  # rose
    "#44CFC8",  # teal
    "#F3AF50",   # wine
    "#C5BFFF",  # olive
    "#11703E"  # green
]
my_params = myplt.my_params(size=12, font_family='Futura', title_size=14)
matplotlib.rcParams.update(my_params)

# %% South Polar map figures 

reload(myplt)
reload(sopo)
def sopolar_classes(class_locs, 
                    ax=None, figsize=(9,9), dotsize=0.5, markerscale=8, dotalpha=0.5,
                    legend=True):
    """ 
    Single plot, all classes by color
    @param class_locs: List of xarray DataArrays with locations for each class
         inds: Classes to plot (default is all classes)
                Reminder that indices are 0-7 for classes 1-8
         """
    if ax is None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = plt.subplot(projection=ccrs.SouthPolarStereo())

    sopo.format_southpolar(ax)

    for cl in range(0, len(class_locs.keys())):
        ax.scatter(class_locs[cl].longitude, class_locs[cl].latitude, alpha=dotalpha, s=dotsize, 
                    transform=ccrs.PlateCarree(), zorder=20,
                    label=('' + str(cl+1)), c=gmm_palette[cl])
    
        ax.add_patch(sopo.fronts_patch('sie')) #sie

    if legend:
        ax.legend(loc='upper right', markerscale=markerscale)
    
    return ax


res_delta_tag = 'Residual Δ-pCO$_{2}$ (µatm)'

def sopolar_classes_paneled(class_locs, axs=None, figsize=(15,10), numpanels=[2,4], 
                            map_facecolor='w',
                            colorProbs=False, 
                            colorError=False, error_param = 'val_error', vlim=None, 
                            add_colorbar=False, dotsize=2, dotalpha=0.8):
    """ 
    Subpaneled plot of all class locations
    @param      class_locs: Dict of xarray Datasets with locations for each class
                            If coloring by probability, pass class_probs (dict of DataFrames)
                colorbyProbs: (default False) Color points by probability ()
                numpanels: [nrows, ncols] for subplots
                            Default is [2,4] for 8 classes. Use [1,8] for one row
    """
    if axs is None:
        fig, axs = plt.subplots(numpanels[0], numpanels[1], figsize=(15,10), layout='constrained', subplot_kw={'projection': ccrs.SouthPolarStereo()})

    for ax in axs.flatten():
        sopo.format_southpolar(ax, map_facecolor = map_facecolor)
        sopo.add_frontlines(ax)

    for ind, ax in enumerate(axs.flatten()[:len(class_locs.keys())]):
        knum = ind # starts at 0
        if str(list(class_locs.keys())[0]) == '1':
            knum = ind + 1
        ax.scatter(class_locs[knum].longitude, class_locs[knum].latitude, 
                alpha=dotalpha, s=dotsize, 
                transform=ccrs.PlateCarree(), 
                label=('' + str(knum)), c=gmm_palette[ind])
        if colorProbs:
            sca = ax.scatter(class_locs[knum].longitude, class_locs[knum].latitude, c=class_locs[knum].probability,
            cmap=cmo.thermal, vmin=0, vmax=1,
            alpha=1, s=dotsize, transform=ccrs.PlateCarree(), label=('class' + str(knum)))
        elif colorError:
            sca = ax.scatter(class_locs[knum].longitude, class_locs[knum].latitude, c=class_locs[knum][error_param],
                cmap='RdBu_r', vmin=-vlim, vmax=vlim,


                alpha=1, s=dotsize, transform=ccrs.PlateCarree(), label=('class' + str(knum)))
        
        ax.set_title('Cluster ' + str(knum), fontsize=14)

    if add_colorbar:
        plt.colorbar(sca, ax=axs.flatten()[:], orientation='horizontal', location='bottom', shrink=0.4,
                      label = res_delta_tag)
    return axs

        

# %% Plotting average profiles by class

def mean_tracer_profiles(classDS, var='CT', 
                         ax=None, figsize=(9,6),
                         shadecolor = gmm_palette[0]):
    """ 
    Plot single tracer profile with shading for std deviation

    @param classDS: xarray Dataset with full profile data (for each class)
                    dims: profid, pressure
                    coords: profid, pressure...
            var: tracer to plot over depth
            
    """
    if ax is None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = fig.gca()

    centerline = classDS.mean(dim='profid')
    lbuffer = centerline - classDS.std(dim='profid')
    rbuffer = centerline + classDS.std(dim='profid')

    ax.scatter(centerline[var], centerline.pressure, c=shadecolor, s=2, alpha=0.5)
    ax.fill_betweenx(centerline.pressure, lbuffer[var], rbuffer[var], color=shadecolor, alpha=0.2)

    ax.set_ylabel('Pressure [dbar]')
    ax.set_xlabel(var)
    ax.grid(alpha=0.3, zorder=0)
    ax.invert_yaxis()

def mean_tracer_profiles_paneled(class_data, vars=['CT', 'SA'],
                                 figsize=(15,7), numpanels = [2,6]):
    """ 
    Paneled by class
    Use mean_tracer_profiles to plot CT and SA for each class
    """
    fig, axs = plt.subplots(numpanels[0], numpanels[1], figsize=figsize, layout='tight', sharey=True)

    for ind, ax in enumerate(axs.flatten()[:numpanels[1]]):
        mean_tracer_profiles(class_data[ind+1], var=vars[0], 
                             ax=ax, shadecolor=gmm_palette[ind])
        ax.set_title('Class ' + str(ind+1) + ' ' + vars[0], fontsize=14)

    for ind, ax in enumerate(axs.flatten()[numpanels[1]:]):
        mean_tracer_profiles(class_data[ind+1], var=vars[1], 
                             ax=ax, shadecolor=gmm_palette[ind])
        ax.set_title('Class ' + str(ind+1) + ' ' + vars[1], fontsize=14)
    
    axs[0,0].invert_yaxis()
    # axs[1,0].invert_yaxis()


# %% Plot GMM posterior probabilities
def boxplot_gmm_probs(class_probs, ax=None, figsize=(9,6), 
                 colors=gmm_palette):
    """ 
    Boxplot of GMM posterior probabilities for each class
    @param class_probs: dict"""
    if ax is None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = fig.gca()

    list = [x.probability.values for x in class_probs.values()]

    bplot = ax.boxplot(list, patch_artist=True, vert=False,
                    tick_labels = [str(int(x)+1) for x in class_probs.keys()], 
                    sym='.',
                    flierprops={'color':'grey', 'markersize':1, 'alpha':0.4, 'marker':'.'},
                    medianprops={'color':'crimson', 'linewidth':1.5})

    # bplot = sns.violinplot(list, alpha=0.4, inner='quart', palette= tol_palette)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(matplotlib.colors.to_rgba(color, alpha=0.5))
        
    ax.set_xlabel('Probability')
    ax.set_ylabel('Class')
    ax.set_yticklabels([str(int(x)+1) for x in class_probs.keys()])
    ax.set_title('GMM posterior probabilities')
    ax.grid(alpha=0.2, zorder=0)
    # ax.set_ylim([0,1])


# %% Time dependence of class probabilities

def scatter_probsXmonth(class_probs, inds=range(8), ax=None, figsize=(9,6), 
                 colors=gmm_palette):
    if ax is None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = fig.gca()
    
    for cl in inds:
        # data = expand_datetime(class_probs[cl], type='dataframe')
        data = class_probs[cl]
        data['doy'] = data['yearday']%365.25
        ax.scatter(data.doy, data.probability, 
                   alpha=0.2, s=4, 
                   label=('' + str(cl+1)), c=colors[cl])
        
    ax.legend(markerscale=7)


def second_probs_seasonality(class_probs, probs, inds=[0,3], ax=None, figsize=(9,6), 
                 colors=gmm_palette):
    """ Sca
    @param:     class_probs: dict of DataFrames with class probabilities
                probs: DataFrame of all class probabilities 
                inds: [cl, nextbest] where cl is the class to plot and nextbest is the class to compare to
                """
    
    if ax is None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = fig.gca()
    
    cl = inds[0]
    nextbest = inds[1]
    data = class_probs[cl]

    temp = probs.loc[data.index] # Table of probabilities for profiles assigned to class cl 
    data['nextcl_'+str(nextbest+1)] = temp.loc[:, nextbest] # Take column of class you want to compare to

    data['doy'] = data['yearday']%365.25
    ax.scatter(data.doy, data.probability, 
                alpha=0.2, s=4, 
                label=('' + str(cl+1)), c=colors[cl])
    
    ax.scatter(data.doy, data['nextcl_'+str(nextbest+1)], 
                alpha=0.2, s=4, 
                label=('' + str(nextbest+1)), c=colors[nextbest+1])
        
    ax.legend(markerscale=7)