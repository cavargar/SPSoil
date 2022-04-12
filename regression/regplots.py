# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score


plotName = 'Performace'

def regPlot(y_test, y_pred, y_test_band, y_pred_band, plotName, save = False):
    r2_band = r2_score(y_test_band , y_pred_band) 
    cc_band = np.corrcoef(y_test_band , y_pred_band)[0,1]

    r2 = r2_score(y_test , y_pred) 
    cc = np.corrcoef(y_test , y_pred)[0,1]

    fig = plt.figure(figsize=(7, 5), dpi=900)
    ax = fig.add_subplot()
    
    
    textstr = '\n'.join((
    'All bands',
    r'  $\rho = $' + str(round(cc,2)),
    r'  $R^2 = $' + str(round(r2,2)),
    'Best band',
    r'  $\rho = $' + str(round(cc_band,2)),
    r'  $R^2 = $'  + str(round(r2_band,2))))
    props = dict(boxstyle='round', facecolor = 'white', alpha=0.1)
    
    plt.text(0.79, 0.29, textstr.rjust(20), fontsize=8,
             transform=plt.gcf().transFigure,
             verticalalignment= 'top', 
             horizontalalignment = 'left',
             bbox=props)

    sns.regplot(x=y_test, y=y_pred, scatter_kws = {'color': 'red', 'alpha': 0.4, 's':6}, line_kws = {'color': 'red', 'alpha': 1, 'lw':1.2}, label = 'All bands')
    sns.regplot(x=y_test_band, y=y_pred_band, scatter_kws = {'color': 'blue', 'alpha': 0.4, 's':6}, line_kws = {'color': 'blue', 'alpha': 1, 'lw':1.2}, label = 'Best band')
    plt.plot(range(-2,3), range(-2,3), ls = '--', color = 'black', label = 'Reference', linewidth = 0.7)
    plt.title(plotName, fontdict = {'fontsize': 10})
    plt.xlabel("True", fontdict = {'fontsize': 9})
    plt.ylabel("Predicted", fontdict = {'fontsize': 9})
    plt.legend(loc=2, fontsize = 8)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 1.5)
    plt.yticks([-0.5,0,0.5,1])
    plt.xticks([0,0.5,1])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 900, quality = 100)
        plt.savefig(svgp, dpi = 900, quality = 100)
        plt.show()

    return


def regPlotBest(y_test, y_pred, y_test_band, y_pred_band, y_test_best, y_pred_best, plotName, save = False):
    r2_band = r2_score(y_test_band , y_pred_band) 
    cc_band = np.corrcoef(y_test_band , y_pred_band)[0,1]

    r2_best = r2_score(y_test_best , y_pred_best) 
    cc_best = np.corrcoef(y_test_best , y_pred_best)[0,1]

    r2 = r2_score(y_test , y_pred) 
    cc = np.corrcoef(y_test , y_pred)[0,1]

    fig = plt.figure(figsize=(7, 5))#, dpi=900)
    ax = fig.add_subplot()
    
    
    textstr = '\n'.join((
    'All features',
    r'  $\rho = $' + str(round(cc,2)),
    r'  $R^2 = $' + str(round(r2,2)),
    'Best feature',
    r'  $\rho = $' + str(round(cc_band,2)),
    r'  $R^2 = $'  + str(round(r2_band,2)),
    '3 Best features',
    r'  $\rho = $' + str(round(cc_best,2)),
    r'  $R^2 = $'  + str(round(r2_best,2))))
    
    props = dict(boxstyle='round', facecolor = 'white', alpha=0.1)
    
    plt.text(0.77, 0.36, textstr.rjust(20), fontsize=8,
             transform=plt.gcf().transFigure,
             verticalalignment= 'top', 
             horizontalalignment = 'left',
             bbox=props)

    sns.regplot(x=y_test, y=y_pred, scatter_kws = {'color': 'red', 'alpha': 0.4, 's':6}, line_kws = {'color': 'red', 'alpha': 1, 'lw':1.2}, label = 'All bands')
    sns.regplot(x=y_test_band, y=y_pred_band, scatter_kws = {'color': 'blue', 'alpha': 0.4, 's':6}, line_kws = {'color': 'blue', 'alpha': 1, 'lw':1.2}, label = 'Best feature')
    sns.regplot(x=y_test_best, y=y_pred_best, scatter_kws = {'color': 'green', 'alpha': 0.4, 's':6}, line_kws = {'color': 'green', 'alpha': 1, 'lw':1.2}, label = '3 Best features')
    plt.plot(range(2,10), range(2,10), ls = '--', color = 'black', label = 'Reference', linewidth = 0.7)
    plt.title(plotName, fontdict = {'fontsize': 10})
    plt.xlabel("True", fontdict = {'fontsize': 9})
    plt.ylabel("Predicted", fontdict = {'fontsize': 9})
    plt.legend(loc=2, fontsize = 8)
    ax.set_xlim(3.5, 8.5)
    ax.set_ylim(3.5, 8.5)
    plt.yticks([4,5,6,7,8])
    plt.xticks([4,5,6,7,8])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 900, quality = 100)
        plt.savefig(svgp, dpi = 900, quality = 100)
        plt.show()

    return

def regPlotBest2(compuesto, y_test, y_pred, y_test_band, y_pred_band, y_test_best, y_pred_best, model_text, plotName, save = False):
    
    r2_band = r2_score(y_test_band , y_pred_band) 
    cc_band = np.corrcoef(y_test_band , y_pred_band)[0,1]

    r2_best = r2_score(y_test_best , y_pred_best) 
    cc_best = np.corrcoef(y_test_best , y_pred_best)[0,1]

    r2 = r2_score(y_test , y_pred) 
    cc = np.corrcoef(y_test , y_pred)[0,1]


    fig = plt.figure(figsize=(7, 5), dpi=500)

    ax = fig.add_subplot()
    
    
    if model_text == 'LASSO':
        bands_legend = 'Lasso selected'    
    else:
        bands_legend = 'All features'
    
    
    if compuesto == 'pH':
        min_lim = 3.9
        max_lim = 8.3
        x_label = 'True'
        y_label = 'Predicted'
    elif compuesto == 'OM':
        min_lim = 0
        max_lim = 20
        x_label = 'True'
        y_label = 'Predicted'
    elif compuesto == 'Ca':
        y_test = np.log10(y_test)
        y_pred = np.log10(y_pred)
        y_test_band = np.log10(y_test_band)
        y_pred_band = np.log10(y_pred_band)
        y_test_best = np.log10(y_test_best)
        y_pred_best = np.log10(y_pred_best)
        x_label =  r' $Log_{10}$(True)'
        y_label = r' $Log_{10}$(Predicted)'
        min_lim = -0.6
        max_lim = 1.9
    elif compuesto == 'Mg':
        min_lim = -0.1
        max_lim = 2.95
        x_label = 'True'
        y_label = 'Predicted'
    elif compuesto == 'K':
        min_lim = 0
        max_lim = 1.1
        x_label = 'True'
        y_label = 'Predicted'
    else:
        min_lim = 0
        max_lim = 1.5
        x_label = 'True'
        y_label = 'Predicted'
               
    textstr = '\n'.join((
    bands_legend,
    r'  $\rho = $' + str(round(cc,2)),
    r'  $R^2 = $' + str(round(r2,2)),
    'Best feature',
    r'  $\rho = $' + str(round(cc_band,2)),
    r'  $R^2 = $'  + str(round(r2_band,2)),
    '3 Best features',
    r'  $\rho = $' + str(round(cc_best,2)),
    r'  $R^2 = $'  + str(round(r2_best,2))))
    
    props = dict(boxstyle='round', facecolor = 'white', alpha=0.1)
    
    plt.text(0.75, 0.36, textstr.rjust(20), fontsize=8,
             transform=plt.gcf().transFigure,
             verticalalignment= 'top', 
             horizontalalignment = 'left',
             bbox=props)

    sns.regplot(x=y_test, y=y_pred, scatter_kws = {'color': 'red', 'alpha': 0.4, 's':6}, line_kws = {'color': 'red', 'alpha': 1, 'lw':1.2}, label = bands_legend)
    sns.regplot(x=y_test_band, y=y_pred_band, scatter_kws = {'color': 'blue', 'alpha': 0.4, 's':6}, line_kws = {'color': 'blue', 'alpha': 1, 'lw':1.2}, label = 'Best feature')
    sns.regplot(x=y_test_best, y=y_pred_best, scatter_kws = {'color': 'green', 'alpha': 0.4, 's':6}, line_kws = {'color': 'green', 'alpha': 1, 'lw':1.2}, label = '3 Best features')


    plt.plot(range(int(min_lim) - 1,int(max_lim) + 2), range(int(min_lim) - 1,int(max_lim) + 2), ls = '--', color = 'black', label = 'Reference', linewidth = 0.7)
    plt.title(plotName, fontdict = {'fontsize': 10})
    plt.xlabel(x_label, fontdict = {'fontsize': 9})
    plt.ylabel(y_label, fontdict = {'fontsize': 9})
    plt.legend(loc=2, fontsize = 8)

    
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 500, quality = 100)
        plt.savefig(svgp, dpi = 500, quality = 100)
        plt.savefig(nameExport + '.pdf', dpi = 900, quality = 100)
        plt.show()

    return



def regPlotBest3(compuesto, y_test, y_pred, y_test_band, y_pred_band, y_test_plsr, y_pred_plsr, model_text, plotName, nc, save = False):    
    fig = plt.figure(figsize=(7, 5), dpi=500)
    ax = fig.add_subplot()
    
    
    if model_text == 'LASSO':
        bands_legend = 'Lasso selected'    
    else:
        bands_legend = 'All features'
    
    
    if compuesto == 'pH':
        min_lim = 3.9
        max_lim = 8.3
        x_label = 'True'
        y_label = 'Predicted'
    elif compuesto == 'OM':
        min_lim = 0
        max_lim = 20
        x_label = 'True'
        y_label = 'Predicted'
    elif compuesto == 'Ca':
        y_test = np.log10(y_test)
        y_pred = np.log10(y_pred)
        y_test_band = np.log10(y_test_band)
        y_pred_band = np.log10(y_pred_band)
        y_test_plsr = np.log10(y_test_plsr)
        y_pred_plsr = np.log10(y_pred_plsr)
        x_label =  r' $Log_{10}$(True)'
        y_label = r' $Log_{10}$(Predicted)'
        min_lim = -0.6
        max_lim = 1.9
    elif compuesto == 'Mg':
        min_lim = -0.1
        max_lim = 2.95
        x_label = 'True'
        y_label = 'Predicted'
    elif compuesto == 'K':
        min_lim = 0
        max_lim = 1.1
        x_label = 'True'
        y_label = 'Predicted'
    else:
        min_lim = 0
        max_lim = 1.5
        x_label = 'True'
        y_label = 'Predicted'
               
    sns.regplot(x=y_test, y=y_pred, scatter_kws = {'color': 'red', 'alpha': 0.4, 's':6.5}, line_kws = {'color': 'red', 'alpha': 1, 'lw':1.2}, label = bands_legend)
    sns.regplot(x=y_test_band, y=y_pred_band, scatter_kws = {'color': 'blue', 'alpha': 0.4, 's':6.5}, line_kws = {'color': 'blue', 'alpha': 1, 'lw':1.2}, label = 'Best feature')
    sns.regplot(x=y_test_plsr, y=y_pred_plsr, scatter_kws = {'color': 'green', 'alpha': 0.4, 's':6.5}, line_kws = {'color': 'green', 'alpha': 1, 'lw':1.2}, label = 'PLSR, ' + str(nc) + ' components')

    plt.plot(range(int(min_lim) - 1,int(max_lim) + 2), range(int(min_lim) - 1,int(max_lim) + 2), ls = '--', color = 'black', label = 'Reference', linewidth = 0.7)
    plt.title(plotName, fontdict = {'fontsize': 12})
    plt.xlabel(x_label, fontdict = {'fontsize': 12})
    plt.ylabel(y_label, fontdict = {'fontsize': 12})
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize = 12)
    
    #tcks = np.arange(int(min(y_test)), max(y_test), 1.5)
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 500, quality = 100)
        plt.savefig(svgp, dpi = 500, quality = 100)
        plt.savefig(nameExport + '.pdf', dpi = 900, quality = 100)
        plt.show()
    return



def regPlotSingle(y_test, y_pred, plotName, plotType = 'PLSR', savePath, save = False):

    r2 = r2_score(y_test , y_pred) 
    cc = np.corrcoef(y_test , y_pred)[0,1]

    fig = plt.figure(figsize=(7, 5), dpi=400)
    ax = fig.add_subplot()
    
    
    textstr = '\n'.join((
    r'  $\rho = $' + str(round(cc,2)),
    r'  $R^2 = $' + str(round(r2,2))))
    props = dict(boxstyle='round', facecolor = 'white', alpha=0.1)
    
    props = dict(boxstyle='round', facecolor = 'white', alpha=0.1)
    
    plt.text(0.75, 0.36, textstr.rjust(20), fontsize=8,
             transform=plt.gcf().transFigure,
             verticalalignment= 'top', 
             horizontalalignment = 'left',
             bbox=props)
    y = y_pred
    min_lim = min(y)
    max_lim = max(y)
    x_label = 'True'
    y_label = 'Predicted'
 
    
    sns.regplot(x=y_test, y=y_pred, scatter_kws = {'color': 'green', 'alpha': 0.4, 's':6}, line_kws = {'color': 'green', 'alpha': 1, 'lw':1.2}, label = plotType)


    plt.plot(range(int(min_lim) - 1,int(max_lim) + 2), range(int(min_lim) - 1,int(max_lim) + 2), ls = '--', color = 'black', label = 'Reference', linewidth = 0.7)
    plt.title(plotName, fontdict = {'fontsize': 10})
    plt.xlabel(x_label, fontdict = {'fontsize': 9})
    plt.ylabel(y_label, fontdict = {'fontsize': 9})
    plt.legend(loc=2, fontsize = 8)

    

    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = nameExport + '.png'
        svgp = nameExport + '.svg'
        plt.savefig(pngp, dpi = 500, quality = 100)
        plt.savefig(svgp, dpi = 500, quality = 100)
        plt.savefig('m_' + svgp)
        plt.savefig(nameExport + '.pdf', dpi = 900, quality = 100)
        plt.show()

    return


def regPlotBest4(compuesto, y_test, y_pred, y_test_band, y_pred_band, y_test_plsr, y_pred_plsr, model_text, plotName, nc, save = False):    
    fig = plt.figure(figsize=(7, 5), dpi=500)
    ax = fig.add_subplot()
    
    
    if model_text == 'LASSO':
        bands_legend = 'Lasso selected'    
    else:
        bands_legend = 'All features'
    
    x_label = 'True'
    y_label = 'Predicted'


               
    sns.regplot(x=y_test, y=y_pred, scatter_kws = {'color': 'red', 'alpha': 0.4, 's':6.5}, line_kws = {'color': 'red', 'alpha': 1, 'lw':1.2}, label = bands_legend)
    sns.regplot(x=y_test_band, y=y_pred_band, scatter_kws = {'color': 'blue', 'alpha': 0.4, 's':6.5}, line_kws = {'color': 'blue', 'alpha': 1, 'lw':1.2}, label = 'Best feature')
    sns.regplot(x=y_test_plsr, y=y_pred_plsr, scatter_kws = {'color': 'green', 'alpha': 0.4, 's':6.5}, line_kws = {'color': 'green', 'alpha': 1, 'lw':1.2}, label = 'PLSR, ' + str(nc) + ' components')

    plt.plot(ls = '--', color = 'black', label = 'Reference', linewidth = 0.7)
    plt.title(plotName, fontdict = {'fontsize': 12})
    plt.xlabel(x_label, fontdict = {'fontsize': 12})
    plt.ylabel(y_label, fontdict = {'fontsize': 12})
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize = 12)
    
    #tcks = np.arange(int(min(y_test)), max(y_test), 1.5)

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = savePath + nameExport + '.png'
        svgp = savePath + nameExport + '.svg'
        plt.savefig(pngp, dpi = 500, quality = 100)
        plt.savefig(svgp, dpi = 500, quality = 100)
        plt.savefig(nameExport + '.pdf', dpi = 900, quality = 100)
        plt.show()
    return
