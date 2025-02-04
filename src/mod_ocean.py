#
""" 
Some repetitive code from mod_cremas, 
A place to hold useful code, perhaps for other projects also!
"""
import os.path
import numpy                 as np
import pandas                as pd
import xarray                as xr
from   datetime              import date, datetime, timedelta                 # for saving figures with today's date
import datetime
import   scipy
import gsw


# %% Time functions
def datetime2ytd(time, ref_time):
    """" Return time in YTD format from datetime format."""
    return (time - np.datetime64(ref_time))/np.timedelta64(1, 'D')

def ytd2datetime(num, ref_time):
    """" Return datetime format to YTD.
    @ param num: (int) number of days since ref_time
      ref_time: (str) reference time in 'YYYY-MM-DD' format
    """
    return (num * np.timedelta64(1,'D')) + np.datetime64(ref_time)

def get_ydsines(yearday):
    """ For adding seasonal variable in Training_RandomForest.ipynb"""
    if (yearday < 0) & (yearday > -365):
        yearday = 365+yearday
    if yearday < -365:
        yearday = 365*2 + yearday
    if yearday >= 365:
        yearday = yearday % 365
    ydcos = np.cos(2*np.pi*np.array(yearday)/365)
    ydsin = np.sin(2*np.pi*np.array(yearday)/365)

    return [ydcos, ydsin]

# %% Spatial functions (from 0.1_SOCAT)

def did_cross_lon180(df):
    """ 
    Returns True if the platform crossed the 180 longitude line,
    i.e. when the longitude delta is large (> 270 degrees)"""
    df = df.copy()
    df.loc[:,'longitude_diff'] = df['longitude'].diff().values
    return (np.abs(df['longitude_diff']) > 270).any()


def get_avg_longitude(df, type='mean'):
    """ 
    Function to fix longitude averaging when crossing longitude 180 to -180
    @param df: dataframe with longitude values
    @return longitude_avg: (float) average longitude value 
    """
    # If any longitude differences are large,
    # Then add 360 to all the negative values of longitude
    # This will work even if the ship crosses the -180 bound several times
    if did_cross_lon180(df):
        index = df[df['longitude'] < 0].index
        df.loc[index, 'longitude'] = df.loc[index, 'longitude'] + 360

    if type == 'mean':
        longitude_avg = df['longitude'].mean()
    elif type == 'median':
        longitude_avg = df['longitude'].median()

    # If the average longitude is greater than 180, 
    # then subtract 360 (full revolution)
    if longitude_avg > 180:
        longitude_avg = longitude_avg - 360

    return longitude_avg 

# %% T-S diagrams and binning functions 

def setup_TS_contours(df, ax=None):
    """  
    For plotting; add T-S contours to a given axis.
    @param: df: dataframe with 'CT' and 'SA' as columns

    Example call for anyfloat (Dataframe):
    # crplot.setup_TS_contours(anyfloat, axs[0])
    # ax.scatter(anyfloat.SA, anyfloat.CT, c='r', s=2, alpha=0.5)
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
    ax.clabel(CS, fmt='%1.2f')
    ax.set_xlabel('SA')
    ax.set_ylabel('CT')

    return ax



def TSbin(df, nbins):
    """ 
    Used in coords_TSbin function
    """
    nobs, bin_temp = np.histogram(df.CT, nbins)
    nobs, bin_sal = np.histogram(df.SA, nbins)   
    return [bin_temp, bin_sal]

def coords_TSbin(df, nbins): 
    """ Manually give bin coordinates to each obs. row in the dataframe
    Allows you to plot more complex quantities in T-S space. 
    """

    [bin_temp, bin_sal] = TSbin(df, nbins)
    # initialize empty rows which will hold new bin coordinates for T and S
    coordT = np.empty(len(df))
    coordS = np.empty(len(df))

    df = df.sort_values(by='CT')
    dfind = []  # find index limits where observations fall into each bin
    for i in range(len(bin_temp)): 
        dfind.append(np.searchsorted(df.CT, bin_temp[i], side='right'))
    for i in range(len(bin_temp)-1):  # notice -1 = nbins
        coordT[dfind[i]:dfind[i+1]] = i
    df['y_temp'] = coordT.astype(int)


    df = df.sort_values(by='SA')
    dfind = []
    for i in range(len(bin_sal)):
        dfind.append(np.searchsorted(df.SA, bin_sal[i], side='right'))
    for i in range(len(bin_sal)-1): 
        coordS[dfind[i]:dfind[i+1]] = i
    df['x_sal'] = coordS.astype(int)

    df = df.sort_values(by=['profid', 'pressure'])
    return df

def array_TSbin(df, nbins, var='oxygen', stat='count'):
    """" Calculates value for the TS-binned array.
    Note that coordT will be stored as "y_temp" and corresponds to row in the array."""
    arr = [ [np.NaN for i in range(nbins)] for j in range(nbins) ]
    for r in range(nbins):
        for c in range(nbins):
            subdf = df[(df['x_sal']==c) & (df['y_temp']==r)]

            if stat == 'mean':
                arr[r][c] = np.nanmean(subdf[var])
            elif stat == 'variance':
                arr[r][c] = np.var(subdf[var])
            elif stat == 'count':
                arr[r][c] = len(subdf[var])
            # change line here 

    return arr


# def TS_contours(df, nbins, sminus=0.23, splus=0.2, tminus=1.5, tplus=0.6, type='density'):
#     """ 
#     @ return: gridvals  - 2D array of density or spice values
#     # To plot in figure, use the following line: 
#     # CS = ax.contour(si,ti,dens, linestyles='dashed', colors='k', alpha=0.4, zorder=3)
#     """
#     # Add density contours
#     # Figure out boudaries (mins and maxs)
#     smin = df.SA.min() - sminus
#     smax = df.SA.max() + splus

#     tmin= df.CT.min() - tminus
#     tmax = df.CT.max() + tplus # 0.5 for df659

#     # Calculate how many gridcells we need in the x and y dimensions
#     xdim = int(round((smax-smin)/0.1+1,0))
#     ydim = int(round((tmax-tmin)+1,0))
    
#     # Create empty grid of zeros
#     gridvals = np.zeros((ydim,xdim))
    
#     # Create temp and salt vectors of appropiate dimensions
#     ti = np.linspace(1,ydim-1,ydim)+tmin
#     si = np.linspace(1,xdim-1,xdim)*0.1+smin

#     # Loop to fill in grid with densities
#     for j in range(0,int(ydim)):
#         for i in range(0, int(xdim)):

#             if type == 'density':
#                 gridvals[j,i]=gsw.sigma0(si[i],ti[j])
#             elif type == 'spice':
#                 gridvals[j,i]=gsw.spiciness0(si[i],ti[j])

    

#     return si, ti, gridvals


