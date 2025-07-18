import matplotlib
import matplotlib.pyplot     as plt 
import matplotlib.path       as     mpath    
import cartopy.feature       as     cfeature     
import cartopy.crs           as     ccrs    

import numpy as np
from importlib import reload

import mod_plotting as myplt

import shapefile


def format_southpolar(ax, 

    max_latitude:          float = -35,
    add_gridlines:         bool  = True,
    color_land:            bool  = True,
    land_edgecolor:        str   = 'grey',
    land_facecolor:        str   = 'grey',
    fontsize:              float = 10,
    map_facecolor:         str   = 'white',
    coast_linewidth:       float = 0.3,
    gridlines_linewidth:   float = 0.5,
    gridlines_color:       str   = 'grey',
    gridlines_alpha:       float = 0.5,
    longitude_label_color: str   = 'grey',
    latitude_label_color:  str   = 'grey'
) -> None:
    """
    @param:
        ax:      Axes object, should have a SouthPolarStereo projection
    """
    
    ### Limit the map to max latitude and below.
    ax.set_extent([-180, 180, -90, max_latitude], ccrs.PlateCarree())  # set to -29.4 for map out to 30 degrees or -39.4 for map out to 40 degrees
   
    ### Tune the subplot layout
    # fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)
    
    ### Make the background of the plot white
    ax.set_facecolor(map_facecolor)
    plot_circle_boundary(ax)


    ### Add gridlines (if True)
    if add_gridlines:
        ax.gridlines(color = gridlines_color, alpha = gridlines_alpha, linewidth = gridlines_linewidth)
        
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

        
    ### Add in coastlines/features
    if color_land:
        ax.add_feature(cfeature.LAND, zorder=1, linewidth = coast_linewidth, edgecolor=land_edgecolor, facecolor=land_facecolor)
    else:
        ax.coastlines(resolution = "50m", zorder=1, linewidth = coast_linewidth)

def add_frontlines(ax, types = ['stf', 'saf', 'pf', 'sacc', 'sie']):
    for type in types:
        ax.add_patch(fronts_patch(type))

def fronts_patch(type='stf'):
    so_fronts = shapefile.Reader('./shapefiles/fronts/so_fronts.shp') 
    stf_mod   = shapefile.Reader('./shapefiles/fronts/stf_mod/stf_mod.shp')

    if 'stf' == type:
        stf  = stf_mod.shape(0).points
        result  = plt.Polygon(stf,  fill=False, edgecolor='grey',   zorder=15)
    if 'saf' == type:
        saf  = so_fronts.shape(1).points
        result  = plt.Polygon(saf,  fill=False, edgecolor='grey',   zorder=14)
    if 'pf' == type:
        pf   = so_fronts.shape(2).points
        result   = plt.Polygon(pf,   fill=False, edgecolor='grey',    zorder=13)
    if 'sacc' == type:
        sacc = so_fronts.shape(3).points
        result = plt.Polygon(sacc, fill=False, edgecolor='grey',  zorder=12)
    if 'sie' == type:
        sie  = so_fronts.shape(4).points
        result  = plt.Polygon(sie,  fill=True,  edgecolor='grey',   zorder=0,  facecolor='darkgrey', alpha=0.4)
    return result


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
