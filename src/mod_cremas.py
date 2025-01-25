import os.path

import numpy                 as np
import pandas                as pd


# Southern Oceans SOCAT data
# socatv2024_SO = pd.read_csv('/Users/sangminsong/Library/CloudStorage/OneDrive-UW/Code/CREMAS/MATLAB-SOCAT/socatv2024_SouthernOceans_DATA.csv')
# socatv2024_SO_INFO = pd.read_csv('/Users/sangminsong/Library/CloudStorage/OneDrive-UW/Code/CREMAS/MATLAB-SOCAT/socatv2024_SouthernOceans_INFO.csv')



# socat_3h = pd.read_csv('../data/SOCATv2024_SO_resampled_3h.csv')


# # %% Time functions
# def datetime2ytd(time, ref_time = '2014-01-01'):
#     """" Return time in YTD format from datetime format."""
#     return (time - np.datetime64(ref_time))/np.timedelta64(1, 'D')

# def ytd2datetime(num, ref_time = '2014-01-01'):
#     """" Return datetime format to YTD."""
#     return (num * np.timedelta64(1,'D')) + np.datetime64(ref_time)

# def get_ydsines(yearday):
#     """ For adding seasonal variable in Training_RandomForest.ipynb"""
#     if (yearday < 0) & (yearday > -365):
#         yearday = 365+yearday
#     if yearday < -365:
#         yearday = 365*2 + yearday
#     if yearday >= 365:
#         yearday = yearday % 365
#     ydcos = np.cos(2*np.pi*np.array(yearday)/365)
#     ydsin = np.sin(2*np.pi*np.array(yearday)/365)

#     return [ydcos, ydsin]



# %% Data functions from Channing Prend (cprend@uw.edu)

def smooth(y, box_pts):
    """ Smooth the data using a boxcar filter (running mean)
    cprend@uw.edu
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def interpolate(x_int, xvals, yvals):
    """ 
    Interpolate the data onto the standard depth grid given by x_int
    cprend@uw.edu"""
    yvals_int = []
    for n in range(0, len(yvals)):
        yvals_int.append(np.interp(x_int, xvals[n, :], yvals[n, :]))
    # convert the interpolated data from a list to numpy array
    return np.asarray(yvals_int)


def integrate(zi, data, depth_range):
    """
    Calculate the vertically integrated data column inventory using the composite trapezoidal rule
    cprend@uw.edu
    """
    n_profs   = len(data)
    zi_start  = abs(zi - depth_range[0]).argmin()
    zi_end    = abs(zi - depth_range[1]).argmin()
    zi_struct = np.ones((n_profs, 1)) * zi[zi_start : zi_end]
    data      = data[:, zi_start : zi_end]
    col_inv   = []
    
    for n in range(0, len(data)):
        col_inv.append(np.trapz(data[n,:][~np.isnan(data[n,:])], zi_struct[n,:][~np.isnan(data[n,:])]))
    return col_inv


def delete_rep(data):
        """
        Define a function that gets rid of repeated values
        cprend@uw.edu  
        """
        vals, inverse, count = np.unique(data, return_inverse=True,
                              return_counts=True)

        idx_vals_repeated = np.where(count > 1)[0]
        vals_repeated = vals[idx_vals_repeated]

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:]) # res gives the indices of the repeated values
    
        for n in range(len(res)): 
            data[res[n-1]]=np.nan # set the repeated values to nans
        return data