# os tools


# % load_ext autoreload
# % autoreload 2 
import os.path
import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date                 # for saving figures with today's date
import datetime
import scipy

# for all plots
import matplotlib
import matplotlib.pyplot     as plt                      # needed to make map setup
import matplotlib.colors     as mpcolors
from   matplotlib.ticker     import EngFormatter         # for degree symbol in axis
import cmocean                                           # to add colorbars
from   cmocean               import cm as cmo

import pyproj  
import geopandas             as     gpd                  # for adding shapefiles of frontal zones 
from   osgeo                 import gdal

# for boxenplots
import seaborn               as     sns
from   scipy.stats           import kruskal              # for boxenplot stats
from   scipy.stats           import mannwhitneyu         # for split violin plot stats
import gsw


def my_params(size=12, font_family='Futura', title_size = 14):
    """ 
    Use: plt.rcParams.update(mod_plot.my_params(size=16))
    """
    plt.style.use('default')
    params = {'legend.fontsize': size, 
            'xtick.labelsize':size, 
            'ytick.labelsize':size, 
            'font.size':size,
            'font.family':font_family,
            'mathtext.fontset':'stixsans',
            'mathtext.bf':'STIXGeneral:bold',
            'axes.titlesize': title_size,
            'figure.titlesize': title_size,}
    return params


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
    return tol_palette

# %% Set up units  ======================================

umol_unit = (r'$\mathbf{[\mu} \mathregular{mol~kg} \mathbf{^{-1}]}$')
uatm_unit = (r'$\mathbf{[\mu} \mathregular{atm}}$')
# umol_unit = (r'$[\mu \mathregular{mol~kg^{-1}}]$')
eke_unit = (r'$\mathbf{[m^2~s^{-2}]}$')
umol_unit_squared = (r'$\mathbf{[\mu} \mathregular{mol^2~kg} \mathbf{^{-2}]}$')
degree_symbol = u'\u00B0'

sigma_unit = (r'$\mathbf{\sigma_0}$ ' + '[kg' + r'$\mathbf{~m^{-3}]}$')
# spice_unit = (r'$ \mathbf{\tau} $ ' + '$\mathbf{[m^{-3}~}$' + 'kg]') #dimensionless
backscatter_unit = ('log([m'  + r'$ \mathbf{^{-1}} $' + '])')
par_unit = ('[W m' + r'$ \mathbf{^{-1}} $' + ']')
hb_unit = ('[s' + r'$ \mathregular{^{-2}} $' + ']')
fsle_unit =  ('[days ' + r'$\mathbf{^{-1}}$' + ']')
sa_unit = ('S$_\mathregular_A}$ [g kg ' + r'$\mathbf{^{-1}}$' + ']')
ct_unit = (r'$ \mathbf{\Theta} $' + ' [' + degree_symbol + ' C]')


delta_title = (r'$\mathbf{\Delta }\mathregular{N_{ML}}$ ')
overline_title = (r'$\overline{\mathregular{N}} \mathregular{_{ML}}$ ')
hvar_title = ('s'+ r'$ \mathbf{^2_{H,NO_3}}$ ')
bbp_title = ('bbp' + r'$_{\mathregular{470}} $')
hb_title = ('|' + r'$\mathbf{\nabla_h}\mathregular{b}$' + '|')

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


# %% Functions

# %% T-S Diagrams  ======================================

def setup_TS_contours(df, ax=None, contour_font_size=12):
    """  
    Add T-S contours to a given axis.
    @param: df: dataframe with 'CT' and 'SA' as columns
    """

    if ax == None:
        fig = plt.figure(figsize=(9,7))
        ax = fig.gca()

    # Add density contours
    # Figure out boudaries (mins and maxs)
    smin = df.SA.min() -.2
    smax = df.SA.max() +.2

    tmin= df.CT.min() - 1.2
    tmax = df.CT.max() + 0.5

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax-smin)/0.1+1,0))
    ydim = int(round((tmax-tmin)+1,0))
    
    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))
    
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.sigma0(si[i],ti[j])

    CS = ax.contour(si,ti,dens, linestyles='dashed', colors='k', alpha=0.4, zorder=1)
    ax.clabel(CS, fmt='%1.2f', fontsize=contour_font_size)
    ax.set_xlabel('SA')
    ax.set_ylabel('CT')

    return ax


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base       = matplotlib.colormaps.get_cmap(base_cmap)
    
    color_list = base(np.linspace(0, 1, N))
    cmap_name  = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


# %% Error analysis
def single_boxplot(data, ax=None, boxcolor='r', lw=1.5, flieralpha=0):
    """ 
    @param
    data: cv_kfold.val_error['Model_G'].values # list of 10MAEs
    """
    if ax == None:
        fig  = plt.figure(figsize=(6,2), tight_layout=True)
        ax = plt.gca()

    # lw= 1.5
    bplot = ax.boxplot(data, widths=0.35, vert=False, patch_artist=True,
                    medianprops= {'color':boxcolor, 'linewidth':lw},
                        capprops={'color':boxcolor, 'linewidth':lw},
                        whiskerprops={'color':boxcolor, 'linewidth':lw},
                        flierprops={'markeredgecolor': 'gray', 'marker':'|', 'alpha':flieralpha, 'zorder':1}, #{'color':boxcolor, 'linewidth':1.5},
                        boxprops = {'color':boxcolor, 'linewidth':lw})
    
    ax.grid(zorder=1, alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylabel('', fontsize=16)
    # ax.set_yticklabels([''])
    ax.set_ylim([0.75, 1.25])


    colors = [boxcolor]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.3))

    return ax

# Used to be called boxplot_across_folds
def boxplot_across_columns(data, ax=None, boxcolor='r'):
    """ 
    @param
    if data has columns, each is a different line
    data: cv_kfold.val_error['Model_G'].values # list of 10MAEs
    """
    if ax == None:
        fig  = plt.figure(figsize=(6,4), tight_layout=True)
        ax = plt.gca()

    lw= 1.5
    bplot = ax.boxplot(data, widths=0.65, vert=False, patch_artist=True,
                    medianprops = {'color':'k', 'linewidth':lw},
                    capprops= {'color':'k', 'linewidth':lw},
                    flierprops= {'color':'k', 'linewidth':lw},
                    boxprops = {'color':'k', 'linewidth':lw})
    
    # ax.set_yticklabels([x[-1] for x in data.keys()])
    ax.grid(zorder=1, alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylabel('', fontsize=16)
    # ax.set_yticklabels([''])
    # ax.set_xlabel("Nitrate Error " + umol_unit)

    # ax.axvline(x=0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)
    # ax.axvline(x=-0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)
    ax.axvline(x=0, color='k', linestyle='dotted', linewidth=1.5, alpha=0.7, zorder=0)

    colors = [boxcolor]
    # for mdl in model_list:
    #     colors.append(model_palettes[mdl])

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.5))
    return ax


def error_kde(data, ax=None, textsize=14, ymax=None, pltcolor='r', 
              linelabel='', linestyle='solid', lw=2):
    """
    New KDE plot using all combined validation errors from K-Fold 
    @param:     data       list of errors

    """ 

    if ax == None:
        fig  = plt.figure(figsize=(6,4), tight_layout=True)
        ax = plt.gca()

    # Add Gaussian KDE to estimate probability density function
    x = np.linspace(data.min(), data.max(), 1000)
    kde = scipy.stats.gaussian_kde(data)

    ls = linestyle
    den = ax.plot(x, kde(x), color=pltcolor, linewidth=lw, linestyle=ls, alpha=0.6, label=linelabel)

    if ymax != None:
        ax.set_ylim([0, ymax])

    sns.set_palette('Dark2')
    ax.grid(alpha=0.5, zorder=1)
    ax.axvline(x=0, color='k', linestyle='dotted', linewidth=1.5, alpha=0.7, zorder=0)

    if len(linelabel)>0:
        leg = ax.legend(fontsize=14, framealpha=1)
        # for legobj in leg.legend_Handles:
        #     legobj.set_linewidth(3.5)

    return ax


# %% Maps
def plot_histogram_of_profile_locations(ploc, profiles, lon_range, lat_range,
                                        source='all', binsize=2,
                                        bathy_fname="bathy.nc",
                                        lev_range=range(-6000,1,500),
                                        myPlotLevels=30, vmin=0, vmax=200):
## Histogram of profile logations

# argo_ave = argodat_DS.mean(dim='pressure')
    # binsize=2
    # lev_range=range(-6000,1,500)
    # myPlotLevels=30
    # vmin=0
    # vmax=100

    lon_bins = np.arange(lon_range[0], lon_range[1], binsize)
    lat_bins = np.arange(lat_range[0], lat_range[1], binsize)
    # histogram


    plt.figure(figsize=(20, 13))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                    ccrs.PlateCarree())

    # colormesh histogram

    ### using numpy.histogram2d
    bin_values,_,__ = np.histogram2d(argo_ave.longitude.values,argo_ave.latitude.values,bins=(lon_bins, lat_bins) )
    X, Y = np.meshgrid(lon_bins, lat_bins)

    ax.hist2d(argo_ave.longitude.values,argo_ave.latitude.values,bins=(lon_bins, lat_bins), vmax=vmax, cmap=cmo.amp)


    ax.coastlines(resolution='50m',color='white')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='gray', alpha=0.6, linestyle='--')
    # ax.add_feature(cartopy.feature.LAND)

    ax.add_feature(cfeature.LAND, zorder=16, linewidth = coast_linewidth, edgecolor = land_edgecolor, facecolor = land_facecolor)

    # separate colorbar
    a = np.array([[vmin,vmax]])
    plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap=cmo.amp)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    cbar = plt.colorbar(orientation="horizontal", cax=cax)
    cbar.ax.tick_params(labelsize=22)



