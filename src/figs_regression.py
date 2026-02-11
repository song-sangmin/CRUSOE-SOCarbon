import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors     as mpcolors
import cartopy.crs as ccrs

import mod_plotting as mod_plot
import mod_regression as mod_reg
import figs_pcm
from scipy import stats
from mod_ocean import expand_datetime

def singleRun_error_boxplot(singleRun, byClass = False, ax=None, 
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
        for k in range(1,len(singleRun.ind_list)+1):
            plot_data[k] = singleRun.DF_err[k].val_error.values
        bplot = ax.boxplot(plot_data.values(),
                    widths=0.65, vert=False, patch_artist=True,
                    medianprops = {'color':'k', 'linewidth':lw},
                    capprops= {'color':'k', 'linewidth':lw},
                    flierprops= {'color':'k', 'marker':'|', 'linewidth':lw, 'alpha':0.6, 'zorder':1},
                    boxprops = {'color':'k', 'linewidth':lw})
        # for patch, color in zip(bplot['boxes'], boxcolor):
        #     patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.4))

    else: # single boxplot, all classes combined
        if ax is None:
            fig = plt.figure(figsize=(7,2)); ax = fig.gca()
    
        plot_data = pd.concat([singleRun.DF_err[x] for x in range(1,len(singleRun.ind_list)+1)], axis=0)
        bplot = ax.boxplot(plot_data.val_error.values,
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


def singleRun_error_kde(singleRun, byClass=False, ax=None, xlim=50):
    """ 
    Plot error KDE for a single model run 
    @param     singleRun: ModelVersion object with ind_list corresponding to class number
                byClass:  True to overlay separate KDE's from each class 
    """
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca()
        
    if byClass:
        errors_dict = {k:df for k, df in singleRun.weighted_validation.groupby('cluster')}
        # Add all the models, overlaid
        plot_data = {k:None for k in singleRun.ind_list}
        for k in singleRun.ind_list:
            # plot_data[k] = singleRun.DF_err[k].val_error.values
            plot_data[k] = errors_dict[k].val_error.dropna().values

        colors = sns.color_palette("Dark2", n_colors=10)
        for ind, k in enumerate(singleRun.ind_list):
            mod_plot.error_kde(plot_data[k], ax=ax, linelabel='class'+str(k), 
                            pltcolor=colors[ind], linestyle='solid')
        ax.legend(fontsize=12, loc='right')

    ax.set_xlim([-xlim, xlim])

    # Add the overall 
    plot_data_comb = singleRun.weighted_validation.val_error.values
    sns.set_palette('Dark2')
    mod_plot.error_kde(plot_data_comb, ax=ax, linelabel='combined', pltcolor='k', linestyle='solid')
    # mod_plot.error_kde(plot_data[run_tag], ax=ax, linelabel=run_tag)

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
    # mod_plot.error_kde(plot_data[run_tag], ax=ax, linelabel=run_tag)

    # Add all the rest
    colors = sns.color_palette("Dark2", n_colors=len(run_tags))
    for ind, run_tag in enumerate(run_tags[:]):
        if 'float' in run_tag: ls= 'dashed'
        elif 'ship' in run_tag: ls= 'dotted'
        else: ls= 'solid'

        if show_legend:
            mod_plot.error_kde(plot_data[run_tag], ax=ax, 
                            linelabel=run_tag.replace('-delta_pco2',''), 
                            pltcolor=colors[ind], linestyle=ls)
        else:
            mod_plot.error_kde(plot_data[run_tag], ax=ax, 
                            pltcolor=colors[ind], linestyle=ls)

    ax.legend(fontsize=12, loc='right')
    # ax.set_xlim([-60,60])
    ax.set_xlim([-xlim, xlim])

    


    return ax



# %% Monthly distributions


def error_histplot_paneled_monthly(error_sets=[], axs=None, nbins=30, stat_type='frequency', zero_line=True, error_lim = 100, colors=None, alphas=[]):
    if axs is None:
        fig, axs = plt.subplots(3,4, figsize=(18,9), layout='tight', sharex='col')
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
            sns.histplot(plot_data['val_error'], bins=nbins, kde=True, stat=stat_type, ax=ax, color=colors[ind], alpha=alphas[ind])
            
        
        ax.set_title(monthlist[monthnum-1], fontsize=14)

    
    for ax in axs.flatten():
        if zero_line:
            upperlim = ax.get_ylim()[1]
            ax.vlines(0, ymin=0, ymax=upperlim, colors='r', linewidth=2, linestyles='dashed', alpha=0.6)
            ax.set_ylim([0, upperlim])
        ax.set_xlim([-error_lim,error_lim])
        ax.grid(linestyle='--', alpha=0.5, zorder=0)

    return fig, axs

def plot_decile_calibration(ax, cal_pred, cal_obs, title, stat_var='mean'):
# def plot_decile_calibration(ax, valerrorDF, run_tag):
    # valerrorDF = storedRuns_weighted[run_tag].weighted_validation
    # valerrorDF['n_decile'] = pd.qcut(valerrorDF['weighted_pred'], 10, labels=list(range(1, 11))) #
    # cal_pred = valerrorDF.groupby('n_decile')['weighted_pred'].agg(['mean', 'min', 'max', 'count'])
    # cal_obs = valerrorDF.groupby('n_decile')['delta_pco2'].agg(['mean', 'min', 'max', 'count'])

    ax.scatter(cal_pred[stat_var].values, cal_obs[stat_var].values, color='blue', s=30, label='decile means')
    ax.plot([-100,100], [-100,100], color='black', linestyle='--', alpha=0.5, zorder=1)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.vlines(x=0, ymin=-65, ymax=15, colors='gray', linestyles='-', alpha=0.5)
    ax.hlines(y=0, xmin=-65, xmax=15, colors='gray', linestyles='-', alpha=0.5)
    ax.set_xlim([-65,15])
    ax.set_ylim([-65,15])

    ax.set_title(title) #[4:])
    ax.set_xlabel('estimated')
    ax.set_ylabel('observed')
    
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

