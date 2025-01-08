# os tools

import os.path

import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date                 # for saving figures with today's date
import datetime


# for all plots
import matplotlib
import matplotlib.pyplot     as plt                      # needed to make map setup
import matplotlib.colors     as colors
from   matplotlib.ticker     import EngFormatter         # for degree symbol in axis
import cmocean                                           # to add colorbars
from   cmocean               import cm as cmo


# for map
import shapefile
import cartopy                                           # to make map
import matplotlib.path       as     mpath                # to draw circle for map
import cartopy.crs           as     ccrs                 # for map projection
import cartopy.feature       as     cfeature             # to add land features to map
# from   xhistogram.xarray     import histogram            # for map histogram
# from   mycolorpy             import colorlist as mcp     # to get n colors list


# for MaxEnt plots
import pyproj  
import geopandas             as     gpd                  # for adding shapefiles of frontal zones 
from   osgeo                 import gdal
import scikit_posthocs       as     sp                   # for stats


# for boxenplots
import seaborn               as     sns
from   scipy.stats           import kruskal              # for boxenplot stats
from   scipy.stats           import mannwhitneyu         # for split violin plot stats

# %% Import data for fronts

so_fronts = shapefile.Reader('./shapefiles/fronts/so_fronts.shp') 
stf_mod   = shapefile.Reader('./shapefiles/fronts/stf_mod/stf_mod.shp')

stf  = stf_mod.shape(0).points
saf  = so_fronts.shape(1).points
pf   = so_fronts.shape(2).points
sacc = so_fronts.shape(3).points
sie  = so_fronts.shape(4).points



# %% Functions

##################################################################
######  Set up Southern Ocean Map  ###############################
##################################################################

def setup_SO_axes(
    ax:                    matplotlib.axes.Axes,
    fig:                   matplotlib.figure.Figure,
    max_latitude:          float = -30,
    add_gridlines:         bool  = True,
    color_land:            bool  = False,
    land_edgecolor:        str   = 'grey',
    land_facecolor:        str   = 'grey',
    fontsize:              float = 10,
    map_facecolor:         str   = 'white',
    coast_linewidth:       float = 0.3,
    gridlines_linewidth:   float = 0.5,
    girdlines_color:       str   = 'grey',
    gridlines_alpha:       float = 0.5,
    longitude_label_color: str   = 'grey',
    latitude_label_color:  str   = 'grey'
) -> None:
    """
    Adapted from Hannah Joy Warren (originally "map_southern_ocean_axes_setup()"
    
    This function sets up the subplot so that it is a cartopy map of the Southern Ocean.
    returns void as the ax and figure objects are pointers not data.
    Args:
        ax  (matplotlib.axes.Axes):     The axis object to add the map to.
        fig (matplotlib.figure.Figure): The figure object for the figure in general.
        add_gridlines (bool):           Whether or not to add gridlines to the plot.
    """
    
    
    ### Limit the map to -40 degrees latitude and below.
    ax.set_extent([-180, 180, -90, max_latitude+0.6], ccrs.PlateCarree())  # set to -29.4 for map out to 30 degrees or -39.4 for map out to 40 degrees
   
    ### Tune the subplot layout
    # fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)
    
    ### Make the background of the plot white
    ax.set_facecolor(map_facecolor)

    ### Make SO plot boundary a circle
    def plot_circle_boundary() -> None:
        """
        Make SO plot boundary a circle.
        Compute a circle in axes coordinates, which we can use as a boundary for the map.
        We can pan/zoom as much as we like - the boundary will be permanently circular.
        """
        theta  = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5  ## could use 0.45 here, as Simon Thomas did
        verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform = ax.transAxes)

    plot_circle_boundary()


    ### Add gridlines (if True)
    if add_gridlines:
        ax.gridlines(color = girdlines_color, alpha = gridlines_alpha, linewidth = gridlines_linewidth)
        
                # specifying xlocs/ylocs yields number of meridian/parallel lines
        dmeridian = 60  # spacing for lines of meridian
        dparallel = 20  # spacing for lines of parallel -- can change this to 10
        num_merid = int(360/dmeridian + 1)
        num_parra = int(180/dparallel + 1)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                          xlocs=np.linspace(-180, 180, num_merid), 
                          ylocs=np.linspace(-90, 90, num_parra), 
                          linestyle="-", linewidth=0.5, color='grey', alpha=gridlines_alpha)
        
        # for label alignment
        va = 'center' # also bottom, top
        ha = 'center' # right, left
        degree_symbol = u'\u00B0'

        # for locations of (meridional/longitude) labels
        lond = np.linspace(-180, 180, num_merid)
        latd = np.zeros(len(lond))

        # for (alon, alat) in zip(lond, latd):
        #     projx1, projy1 = ax.projection.transform_point(alon, max_latitude+1, ccrs.Geodetic())  # set to -29 for map out to 30 degrees or -39 for a map out to 40 degrees
        #     if alon>-180 and alon<0:
        #         ha = 'right'
        #         va = 'center'
        #     if alon>0 and alon<180:
        #         ha = 'left'
        #         va = 'center'
        #     if np.abs(alon-0)<0.01:
        #         ha = 'center'
        #         va = 'bottom'
        #     if alon==-180:
        #         ha = 'center'
        #         va = 'top'
        #     if (alon<180):
        #         txt =  ' {0} '.format(str(int(alon)))+degree_symbol
        #         ax.text(projx1, projy1, txt, va=va, ha=ha, color=latitude_label_color, fontsize=fontsize)
                
        # # for locations of (meridional/longitude) labels select longitude: 315 for label positioning
        # lond2 = 60*np.ones(len(lond))
        # latd2 = np.linspace(-90, 90, num_parra)
        # va, ha = 'center', 'center'
        # for (alon, alat) in zip(lond2, latd2):
        #     projx1, projy1 = ax.projection.transform_point(alon, alat, ccrs.Geodetic())
        #     txt =  ' {0} '.format(str(int(alat)))+degree_symbol
        #     ax.text(projx1, projy1, txt, va=va, ha=ha, color=longitude_label_color, fontsize=fontsize) 
        
        
    ### Add in coastlines/features
    if color_land:
        ax.add_feature(cfeature.LAND, zorder=1, linewidth = coast_linewidth, edgecolor=land_edgecolor, facecolor=land_facecolor)
    else:
        ax.coastlines(resolution = "50m", zorder=1, linewidth = coast_linewidth)

### Make SO plot boundary a circle
def plot_circle_boundary(ax) -> None:
    """
    Make SO plot boundary a circle.
    Compute a circle in axes coordinates, which we can use as a boundary for the map.
    We can pan/zoom as much as we like - the boundary will be permanently circular.
    """
    theta  = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5  ## could use 0.45 here, as Simon Thomas did
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform = ax.transAxes)



def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base       = matplotlib.colormaps.get_cmap(base_cmap)
    
    color_list = base(np.linspace(0, 1, N))
    cmap_name  = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


# %% Example code:

# # Plot of all SOCAT data

# map_proj = ccrs.SouthPolarStereo()

# fig  = plt.figure(figsize=[8,8], dpi=300) # inches
# ax1  = plt.subplot(projection = map_proj)

# # Set up plot axes
# map_southern_ocean_axes_setup(ax1, fig, 
#                               add_gridlines         = False, 
#                               color_land            = True,
#                               land_facecolor        = land_facecolor,
#                               land_edgecolor        = land_edgecolor,
#                               fontsize              = fontsize_small,
#                               map_facecolor         = plot_facecolor, #'#d7dce8',
#                               coast_linewidth       = coast_linewidth)


# ### Add front and sea ice edge
# stf_patch  = plt.Polygon(stf,  fill=False, edgecolor=stf_edgecolor_gmm_map,   zorder=15)
# saf_patch  = plt.Polygon(saf,  fill=False, edgecolor=saf_edgecolor_gmm_map,   zorder=14)
# pf_patch   = plt.Polygon(pf,   fill=False, edgecolor=pf_edgecolor_gmm_map,    zorder=13)
# sacc_patch = plt.Polygon(sacc, fill=False, edgecolor=sacc_edgecolor_gmm_map,  zorder=12)
# sie_patch  = plt.Polygon(sie,  fill=True,  edgecolor=sie_edgecolor_gmm_map,   zorder=0,  facecolor=sie_facecolor_gmm_map, alpha=0.4)

# ax1.add_patch(stf_patch)
# ax1.add_patch(saf_patch)
# ax1.add_patch(pf_patch)
# ax1.add_patch(sacc_patch)
# ax1.add_patch(sie_patch)


# ### Add land on top
# ax1.add_feature(cfeature.LAND, zorder=16, linewidth = coast_linewidth, edgecolor = land_edgecolor, facecolor = land_facecolor)


# for k in expokeys:
#     dat = expo_dict_3h[k]
#     if len(dat) > 0:
#         ax1.scatter(x=dat.longitude.values, y=dat.latitude.values, c='m', transform=ccrs.PlateCarree(), zorder=10,  s=1, alpha=0.3)

# # plot_circle_boundary(ax1)  # this doesn't work to plot the circle again over top


# plt.show()
