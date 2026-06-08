import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors     as mpcolors
import cartopy.crs as ccrs
import numpy as np
import scipy as scipy

import mod_plotting as mod_plot
import mod_regression as mod_reg
import figs_pcm
from scipy import stats
from mod_ocean import expand_datetime

res_delta_tag = 'Residual Δ-pCO$_{2}$ (µatm)'


def error_kde(data, ax=None, textsize=14, ymax=None, pltcolor='r', 
              linelabel='', linestyle='solid', linealpha = 0.7, lw=2):
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
    ax.axvline(x=0, color='k', linestyle='dotted', linewidth=1.5, alpha=linealpha, zorder=0)

    if len(linelabel)>0:
        leg = ax.legend(fontsize=14, framealpha=1)
        # for legobj in leg.legend_Handles:
        #     legobj.set_linewidth(3.5)

    return ax



def singleRun_error_boxplot(singleRun, byClass = False, error_var = 'val_error', ax=None, 
                            xlims=None, lw=1, boxcolor='green'):
    """ 
    note: used to be plot_singlerun_errors
    @ param     singleRun is a storedRun[run_tag] object 
                (ModelVersion in mod_regression)
                byClass: True to plot separate boxplots from each class
    """

    if byClass: # multiple boxplots, one per class
        if ax is None:
            fig = plt.figure(figsize=(7,5)); ax = fig.gca()
        
        plot_data = {}
        errors_dict = {int(k):df for k, df in singleRun.weighted_validation.groupby('cluster')}
        # bplot_dict = {int(k):df for k, df in singleRun.weighted_validation.groupby('cluster')}

        for k in singleRun.ind_list:
            plot_data[int(k)] = errors_dict[k][error_var].dropna().values

        bplot = ax.boxplot(plot_data.values(),
                    widths=0.65, vert=False, patch_artist=True,
                    medianprops = {'color':'k', 'linewidth':lw},
                    capprops= {'color':'k', 'linewidth':lw},
                    flierprops= {'color':'k', 'marker':'|', 'linewidth':lw, 'alpha':0, 'zorder':1},
                    boxprops = {'color':'k', 'linewidth':lw})
        
        for patch in bplot['boxes']:
            patch.set_facecolor(boxcolor)
            patch.set_alpha(0.6)
        # for patch, color in zip(bplot, np.tile(boxcolor, len(singleRun.ind_list))):
        #     patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.4))

    else: # single boxplot, all classes combined
        if ax is None:
            fig = plt.figure(figsize=(7,2)); ax = fig.gca()
    
        # plot_data = pd.concat([singleRun.weighted_validation[x] for x in range(1,len(singleRun.ind_list)+1)], axis=0)
        bplot = ax.boxplot(singleRun.weighted_validation[error_var].values,
                    widths=0.65, vert=False, patch_artist=True,
                    medianprops = {'color':'k', 'linewidth':lw},
                    capprops= {'color':'k', 'linewidth':lw},
                    flierprops= {'color':'k', 'marker':'|', 'linewidth':lw, 'alpha':0, 'zorder':1},
                    boxprops = {'color':'k', 'linewidth':lw})

        for patch, color in zip(bplot['boxes'], boxcolor): 
            patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.4))

    ax.grid(axis='x', linestyle='-', alpha=0.5)
    if xlims != None:
        ax.set_xlim([xlims[0], xlims[1]])
    
    return ax


def singleRun_error_kde(singleRun, byClass=False, error_var = 'val_error', ax=None, xlim=50, linecolor = 'k', linelabel='combined'):
    """ 
    Plot error KDE for a single model run 
    @param     singleRun: ModelVersion object with ind_list corresponding to class number
                byClass:  True to overlay separate KDE's from each class 
    """
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca()
        
    if byClass:
        errors_dict = {int(k):df for k, df in singleRun.weighted_validation.groupby('cluster')}
        # Add all the models, overlaid
        # print('temp')
        plot_data = {k:None for k in singleRun.ind_list}
        for k in singleRun.ind_list:
            # plot_data[k] = singleRun.DF_err[k].val_error.values
            plot_data[k] = errors_dict[k][error_var].dropna().values

        colors = sns.color_palette("Dark2", n_colors=10)
        for ind, k in enumerate(singleRun.ind_list):
            error_kde(plot_data[k], ax=ax, linelabel='Cluster'+str(k), 
                            pltcolor=colors[ind], linestyle='solid')
        ax.legend(fontsize=12, loc='right')

    ax.set_xlim([-xlim, xlim])

    # Add the overall 
    plot_data_comb = singleRun.weighted_validation[error_var].values
    sns.set_palette('Dark2')
    error_kde(plot_data_comb, ax=ax, linelabel=linelabel, pltcolor=linecolor, linestyle='solid')
    # error_kde(plot_data[run_tag], ax=ax, linelabel=run_tag)

    return ax

def storedRuns_error_kde(storedRuns, ax=None, xlim=50, show_legend=True):
    """ 

    @param     storedRuns: dict of ModelVersion objects, keys are run_tags 

    """
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca()

    # PLOT ACROSS RUNS TO SEE BEST PARAMETERS
    run_tags = [k for k in storedRuns.keys()]
    plot_data = {k:None for k in run_tags}

    for run_tag in run_tags[:]:
        singleRun = storedRuns[run_tag]
        plot_data[run_tag] = pd.concat(singleRun.DF_err.values()).val_error.values

    # Single plot
    # run_tag = run_tags[0]
    # fig = plt.figure(figsize=(12,6))
    # ax = fig.gca()
    # error_kde(plot_data[run_tag], ax=ax, linelabel=run_tag)

    # Add all the rest
    colors = sns.color_palette("Dark2", n_colors=len(run_tags))
    for ind, run_tag in enumerate(run_tags[:]):
        if 'float' in run_tag: ls= 'dashed'
        elif 'ship' in run_tag: ls= 'dotted'
        else: ls= 'solid'

        if show_legend:
            error_kde(plot_data[run_tag], ax=ax, 
                            linelabel=run_tag.replace('-delta_pco2',''), 
                            pltcolor=colors[ind], linestyle=ls)
        else:
            error_kde(plot_data[run_tag], ax=ax, 
                            pltcolor=colors[ind], linestyle=ls)

    ax.legend(fontsize=12, loc='right')
    # ax.set_xlim([-60,60])
    ax.set_xlim([-xlim, xlim])

    


    return ax



# %% Monthly distributions


def error_histplot_paneled_monthly(error_sets=[], axs=None, figsize = (18,9), 
                                   nbins=30, binwidth = None, error_param = 'val_error',
                                stat_type='frequency', 
                                zero_line=True, error_lim = 100, colors=None, alphas=[]):
    if axs is None:
        fig, axs = plt.subplots(3,4, figsize=figsize, layout='constrained', sharex='col', sharey='row')
        axs = axs.flatten()

    if colors is None: colors = ['skyblue', 'orange', 'purple']
    monthlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if alphas == []: alphas = [0.8 for x in range(len(error_sets))]

    for monthnum in range(1,13):
        ax = axs[monthnum-1]
        
        for ind, valDF in enumerate(error_sets):
            # histalpha=0.8
            # make sure dataframe has month
            # valDF = expand_datetime(valDF, type='dataframe')
            plot_data = valDF[valDF.month == monthnum]

            if binwidth is None:
                sns.histplot(plot_data[error_param], bins=nbins, kde=True, stat=stat_type, ax=ax, color=colors[ind], alpha=alphas[ind])
            else:
                sns.histplot(plot_data[error_param], binwidth=binwidth, kde=True, stat=stat_type, ax=ax, color=colors[ind], alpha=alphas[ind])
            # ax.hist(plot_data['val_error'], bins=nbins)
            # error_kde(plot_data['val_error'], ax=ax, pltcolor=colors[ind], linestyle='solid', linelabel=None, linealpha=alphas[ind])
        
        ax.set_title(monthlist[monthnum-1], fontsize=24)

    
    for ax in axs.flatten():
        if zero_line:
            upperlim = ax.get_ylim()[1]
            ax.vlines(0, ymin=0, ymax=upperlim, colors='k', linewidth=2, linestyles='dashed', alpha=0.6)
            ax.set_ylim([0, upperlim])
        ax.set_xlim([-error_lim,error_lim])
        ax.grid(linestyle='--', alpha=0.5, zorder=0)

    return fig, axs

def plot_decile_calibration(ax, cal_pred, cal_obs, title, stat_var='mean', axlims = [-65, 15]):
# def plot_decile_calibration(ax, valerrorDF, run_tag):
    # valerrorDF = storedRuns_weighted[run_tag].weighted_validation
    # valerrorDF['n_decile'] = pd.qcut(valerrorDF['weighted_pred'], 10, labels=list(range(1, 11))) #
    # cal_pred = valerrorDF.groupby('n_decile')['weighted_pred'].agg(['mean', 'min', 'max', 'count'])
    # cal_obs = valerrorDF.groupby('n_decile')['delta_pco2'].agg(['mean', 'min', 'max', 'count'])
    ax.set_aspect('equal')
    ax.scatter(cal_pred[stat_var].values, cal_obs[stat_var].values, color='k', s=30) #, label='decile means')
    ax.plot([-1000,1000], [-1000,1000], color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # axlims = [-65, 15]
    ax.vlines(x=0, ymin=axlims[0], ymax=axlims[1], colors='gray', linestyles='-', alpha=0.5)
    ax.hlines(y=0, xmin=axlims[0], xmax=axlims[1], colors='gray', linestyles='-', alpha=0.5)
    ax.set_xlim(axlims)
    ax.set_ylim(axlims)

    ax.set_title(title) #[4:])
    ax.set_xlabel('Estimated')
    ax.set_ylabel('Observed')
    
    lincal = stats.linregress(cal_pred[stat_var].values, cal_obs[stat_var].values)

    # Plot the fitted line using plt.axline
    ax.axline(xy1=(0, lincal.intercept), slope=lincal.slope, color='r', 
            label=f'y={lincal.slope:.2f}x+{lincal.intercept:.2f}')

    ax.legend()


    return ax 



# %% Retrieve plotting dictionary

def split_classes(platDF):
    return {k:v for k, v in platDF.groupby('cluster')}

# def get_plotting_dict(types=['train', 'val'], platforms=['combined', 'float', 'ship']):

# %% Cross-validation data splitting for k-fold

datatype_colors = {'train' : "#332288", 
                   'validation' :  "#ED7F1E", 
                   'combined' : "#524F4F",
                   'train_comb':  "#332288", 
                   'val_comb':  "#6EC3EE", 
                   'train_float': "#882255",  # rose
                   'val_float': "#DDCC77",  # rose
                   'train_ship': "#44AA99",  # dark blue
                   'val_ship': "#CC6677",  # dark blue
                   }

def plot_kfold(cvtainer, n_clusters, nfolds, plat_type='comb'):
    # VISUALIZE TRAIN/VAL COUNTS BY FOLD, BY CLASS
    # === choose which data to restrict to 
    plat_type = 'comb' 
    # plat_type = '_float'
    # plat_type = '_ship'

    trainvar = 'train'; valvar = 'validation'
    if plat_type in ['_float', '_ship']:
        trainvar += plat_type; valvar = valvar[:3] + plat_type

    # n_clusters = n_gmm - len(exclude_nums)
    list_folds = ['fold'+str(k) for k in range(1, nfolds+1)]


    # === plot 
    fig, axs = plt.subplots(1, n_clusters, figsize=(2*n_clusters, 4), sharey='row', layout='constrained')

    for ncluster in range(1,n_clusters+1):
        plot_data = {
            trainvar: [cvtainer.countobs[ftag].loc[ncluster, trainvar] for ftag in list_folds],
            valvar: [cvtainer.countobs[ftag].loc[ncluster, valvar] for ftag in list_folds],
        }

        x = np.arange(len(list_folds))  # the label locations
        width = 0.45  # the width of the bars
        multiplier = 0

        # fig, ax = plt.subplots(layout='constrained')

        ax = axs[ncluster-1]

        for subset_type, nobs in plot_data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, nobs, width, label=subset_type, color=datatype_colors[subset_type])
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # ax.set_xticks(x + width, list_folds)
        ax.set_xlabel('Fold #')
        ax.set_title('Class ' + str(ncluster))

    for ax in axs.flatten():
        ax.set_ylim(0,3000)
    ax.legend()
    axs[0].set_ylabel('Counts')
        # ax.legend(loc='upper left', ncols=3)