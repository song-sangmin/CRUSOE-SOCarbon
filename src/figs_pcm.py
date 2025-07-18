# Main use in 1.1_pca_transform.ipynb

import matplotlib
import matplotlib.pyplot     as plt  
import mod_plotting as myplt
import cartopy.crs           as     ccrs    
from importlib import reload

import mod_southpolarplot as sopo

# tol_palette = myplt.tol8()
gmm_palette = [
    "#332288",  # dark blue   
    "#CC6677",  # rose
    "#DDCC77",  # sand
    "#44AA99",  # teal
    "#117733",  # green
    "#88CCEE",  # light blue
    "#882255",   # wine
    "#999933",  # olive
]

my_params = myplt.my_params(size=12, font_family='Futura', title_size=14)
matplotlib.rcParams.update(my_params)

# %% Main figures 

reload(myplt)
reload(sopo)
def sopolar_classes(class_locs, inds = range(8), 
                    ax=None, figsize=(9,9), 
                    legend=True):
    """ 
    @param class_locs: List of xarray DataArrays with locations for each class
         inds: Classes to plot (default is all classes)
                Reminder that indices are 0-7 for classes 1-8
         """
    if ax is None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = plt.subplot(projection=ccrs.SouthPolarStereo())

    sopo.format_southpolar(ax)

    for cl in inds:
        ax.scatter(class_locs[cl].longitude, class_locs[cl].latitude, alpha=0.4, s=.5, 
                    transform=ccrs.PlateCarree(), 
                    label=('' + str(cl+1)), c=gmm_palette[cl])
    
        ax.add_patch(sopo.fronts_patch('sie')) #sie

    if legend:
        ax.legend(loc='upper right', markerscale=8)


def sopolar_classes_paneled(class_locs,
                    ax=None, figsize=(15,10)):
    
    fig, axs = plt.subplots(2,4, figsize=(15,10), layout='tight', subplot_kw={'projection': ccrs.SouthPolarStereo()})

    for ax in axs.flatten():
        sopo.format_southpolar(ax)
        sopo.add_frontlines(ax)

    for ind, ax in enumerate(axs.flatten()):
        ax.scatter(class_locs[ind].longitude, class_locs[ind].latitude, 
                   alpha=0.2, s=1, 
                   transform=ccrs.PlateCarree(), 
                   label=('' + str(ind)), c=gmm_palette[ind])
        ax.set_title('Class ' + str(ind+1), fontsize=14)

        


# %% Plotting average profiles by class



def mean_tracer_profiles(classDS, var='CT', 
                         ax=None, figsize=(9,6),
                         shadecolor = gmm_palette[0]):
    """ Single group
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
                                 figsize=(15,7)):
    
    fig, axs = plt.subplots(2,8, figsize=figsize, layout='tight', sharey=True)
    for ind, ax in enumerate(axs.flatten()[:8]):
        mean_tracer_profiles(class_data[ind], var=vars[0], 
                             ax=ax, shadecolor=gmm_palette[ind])
        ax.set_title('Class ' + str(ind+1) + ' ' + vars[0], fontsize=14)

    for ind, ax in enumerate(axs.flatten()[8:]):
        mean_tracer_profiles(class_data[ind], var=vars[1], 
                             ax=ax, shadecolor=gmm_palette[ind])
        ax.set_title('Class ' + str(ind+1) + ' ' + vars[1], fontsize=14)
    
    axs[0,0].invert_yaxis()
    # axs[1,0].invert_yaxis()
        