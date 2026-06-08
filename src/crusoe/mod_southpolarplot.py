import matplotlib
import matplotlib.pyplot     as plt 
import matplotlib.path       as     mpath    
import cartopy.feature       as     cfeature     
import cartopy.crs           as     ccrs    

import numpy as np
from importlib import reload

import mod_plotting as myplt

import shapefile

# %% SIMPLEST SOUTH POLAR PLOT 

def map_platformDF(platDF, ax=None, figsize=(9,9), dotsize=0.5, markerscale=8, dotalpha=0.5, dotcolor='r', label='',
                   vmin=None, vmax=None,
                   add_gridlabels = False):

    if ax == None:
        fig = plt.figure(figsize=figsize, layout='constrained')
        ax = plt.subplot(projection=ccrs.SouthPolarStereo())

    format_southpolar(ax)

    if vmin == None:
        ax.scatter(platDF.longitude, platDF.latitude, alpha=dotalpha, s=dotsize, 
                transform=ccrs.PlateCarree(), zorder=10,
                label=(label), c=dotcolor)
    else:           
        ax.scatter(platDF.longitude, platDF.latitude, alpha=dotalpha, s=dotsize, 
                    transform=ccrs.PlateCarree(), zorder=10,
                    label=(label), c=dotcolor, vmin=vmin, vmax=vmax)

    ax.add_patch(fronts_patch('sie')) #sie

    if len(label) >0:
        ax.legend(loc='upper right', markerscale=markerscale)

    return ax

# %% MAIN FORMATTING FUNCTIONS

def format_southpolar(ax, 

    max_latitude:          float = -35,
    add_gridlines:         bool  = True,
    color_land:            bool  = True,
    land_edgecolor:        str   = 'grey',
    land_facecolor:        str   = 'grey',
    fontsize:              float = 10,
    map_facecolor:         str   = 'white',
    coast_linewidth:       float = 0.3,
    gridlines_linewidth:   float = .7,
    gridlines_color:       str   = 'grey',
    gridlines_alpha:       float = 0.5,
    longitude_label_color: str   = 'grey',
    latitude_label_color:  str   = 'grey'
) -> None:
    """
    @param:
        ax:      Axes object, should have a SouthPolarStereo projection
                # fig = plt.figure(figsize=figsize, layout='constrained')
                # ax = plt.subplot(projection=ccrs.SouthPolarStereo())

        """
    
    ### Limit the map to max latitude and below.
    ax.set_extent([-180, 180, -90, max_latitude], ccrs.PlateCarree())  # set to -29.4 for map out to 30 degrees or -39.4 for map out to 40 degrees
   
    ### Tune the subplot layout
    # fig.subplots_adjust(bottom=0.05, top=0.95, left=0.04, right=0.95, wspace=0.02)
    
    ### Make the background of the plot white
    ax.set_facecolor(map_facecolor)
    plot_circle_boundary(ax)

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
    # if add_gridlines:

    #     dmeridian = 60
    #     dparallel = 20

    #     meridians = np.arange(-180, 180 + dmeridian, dmeridian)
    #     parallels = np.arange(-90, max_latitude + dparallel, dparallel)

    #     gl = ax.gridlines(
    #         crs=ccrs.PlateCarree(),
    #         xlocs=meridians,
    #         ylocs=parallels,
    #         linestyle='-',
    #         linewidth=0.5,
    #         color='grey',
    #         alpha=gridlines_alpha
    #     )

    #     degree_symbol = u"\N{DEGREE SIGN}"

    #     # --- Longitude labels (place at outer latitude boundary) ---
    #     for lon in meridians:
    #         ax.text(
    #             lon,
    #             max_latitude,
    #             f"{int(lon)}{degree_symbol}",
    #             transform=ccrs.PlateCarree(),
    #             ha='center',
    #             va='bottom'
    #         )

    #     # --- Latitude labels (place along one meridian, e.g. 0°) ---
    #     for lat in parallels:
    #         if lat > -90:
    #             ax.text(
    #                 0,
    #                 lat,
    #                 f"{int(abs(lat))}{degree_symbol}S",
    #                 transform=ccrs.PlateCarree(),
    #                 ha='left',
    #                 va='center'
    #             )
        
    ### Add in coastlines/features
    if color_land:
        ax.add_feature(cfeature.LAND, zorder=35, linewidth = coast_linewidth, edgecolor=land_edgecolor, facecolor=land_facecolor)
    else:
        ax.coastlines(resolution = "50m", zorder=5, linewidth = coast_linewidth)

    # add sea ice zone
    ax.add_patch(fronts_patch('sie')) #sie

    return ax

def add_frontlines(ax, types = ['stf', 'saf', 'pf', 'sacc', 'sie'], sie_alpha=0.5):
    """ 
    Pass an axis with SouthPolarStereo projection, add specified fronts"""
    for type in types:
        ax.add_patch(fronts_patch(type, sie_alpha))

def fronts_patch(type='stf', sie_alpha=0.5, sie_zorder=1):
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
        result   = plt.Polygon(pf,   fill=False, edgecolor='grey',    zorder=0) #, facecolor='yellow', alpha=0.2)
    if 'sacc' == type:
        sacc = so_fronts.shape(3).points
        result = plt.Polygon(sacc, fill=False, edgecolor='grey',  zorder=1) #, alpha=0.1) #, facecolor='white')
    if 'sie' == type:
        sie  = so_fronts.shape(4).points
        result  = plt.Polygon(sie,  fill=True,  edgecolor='grey',   zorder=sie_zorder,  facecolor='lightgray', alpha=sie_alpha)


    return result


### Make SO plot boundary a circle
def plot_circle_boundary(ax) -> None:
    """
    Make SO plot boundary a circle
    """
    theta  = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5  ## could use 0.45 here, as Simon Thomas did
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform = ax.transAxes)


