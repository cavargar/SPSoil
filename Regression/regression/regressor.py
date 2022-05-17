# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.cross_decomposition import PLSRegression
from joblib import dump, load
import argparse
from cubist import Cubist

parser = argparse.ArgumentParser(description='Pre processing of NIRS data.')
parser.add_argument('-p', '--properties', type=str,
                    help='Space separated list of properties in "quotes", must be contained in properties data set header')
parser.add_argument('-m', '--models', type=str,
                    help='Space separated regression models, available: LR, SVR,LASSO')
parser.add_argument('-f', '--featureM', type=str,
                    help='path feature metrics')
parser.add_argument('-d', '--data', type=str,
                    help='path of pre processed data')



args = parser.parse_args()

def argTolist(argStr, num = False):
    l = []
    for t in argStr.split():
        if num:
            l.append(float(t))
        else:
            l.append(str(t))
    return l


print(args.properties)
print(argTolist(args.properties))
print(args.models)
print(argTolist(args.models))

def regPlotBest(compuesto, y_test, y_pred, y_test_band, y_pred_band, y_test_plsr, y_pred_plsr, model_text, plotName, nc, savePath, save = False):    
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

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    plt.legend(loc=2, fontsize = 8)
    if save:
        nameExport = plotName.replace(' ','_')
        pngp = savePath + nameExport + '.png'
        svgp = savePath + nameExport + '.svg'
        plt.savefig(pngp, dpi = 500)
        #plt.savefig(svgp, dpi = 500)
        #plt.savefig(nameExport + '.pdf', dpi = 900)
        #plt.show()
    return


otrosCompuestos = argTolist(args.properties)
av_models = argTolist(args.models)

#otrosCompuestos = ['pH','OM', 'Ca', 'Mg', 'K','Na']
#otrosCompuestos = ['Ca']

#av_models = ['LR', 'SVR','LASSO']



bestBandMetric = pd.DataFrame({'property':[],
                         'metric':[],
                         'BestBand':[],
                         'Score':[]
                        })
#compuesto='pH'

for compuesto in otrosCompuestos:
    propMetrics = pd.read_csv(args.featureM + 'metrics_' + str(compuesto) + '.csv',sep = ';', decimal = '.')
    
    metrics = list(propMetrics.columns)
    metrics.remove('feature')
    crrntMetric = metrics[4]
    for crrntMetric in metrics:
        tmpPropMetrics = propMetrics.copy()
        tmpPropMetrics.sort_values(by=[crrntMetric], inplace=True, ascending=False, ignore_index = True)
        for c in range(0,3):
            #print(c)
            tmpRow = [compuesto, crrntMetric, tmpPropMetrics['feature'][c], tmpPropMetrics[crrntMetric][c].round(6)]
            bestBandMetric.loc[bestBandMetric.shape[0] + 1] = tmpRow
        


bestBandMetric = bestBandMetric[bestBandMetric.metric == 'Correlation']

# metricsDF = pd.DataFrame({'Property' : [], 'Model':[], 'Best_Band':[], 'PLSR N Compnts':[],
#                           'CC Train All': [], 'CC Test All': [],
#                           'MSE Train All': [], 'MSE CV All': [], 'MSE Test All': [], 
#                           'R2 Train All': [], 'R2 CV All': [], 'R2 Test All': [],
#                           'EV Train All': [], 'EV CV All': [], 'EV Test All': [], 
#                           'CC Train Band': [], 'CC Test Band': [],
#                           'MSE Train Band': [], 'MSE CV Band': [], 'MSE Test Band': [], 
#                           'R2 Train Band': [], 'R2 CV Band': [], 'R2 Test Band': [],
#                           'EV Train Band': [], 'EV CV Band': [], 'EV Test Band': [], 
#                           'CC Train PLSR': [], 'CC Test PLSR': [],
#                           'MSE Train PLSR': [], 'MSE CV PLSR': [], 'MSE Test PLSR': [], 
#                           'R2 Train PLSR': [], 'R2 CV PLSR': [], 'R2 Test PLSR': [],
#                           'EV Train PLSR': [], 'EV CV PLSR': [], 'EV Test PLSR': []
#                           })



compuesto = otrosCompuestos[0]
for compuesto in otrosCompuestos:
    
    metricsDF = pd.DataFrame({'Property' : [], 'Model':[], 'Best_Band':[], 'PLSR N Compnts':[],
                          'CC Train All': [], 'CC Test All': [],
                          'MSE Train All': [], 'MSE CV All': [], 'MSE Test All': [], 
                          'R2 Train All': [], 'R2 CV All': [], 'R2 Test All': [],
                          'EV Train All': [], 'EV CV All': [], 'EV Test All': [], 
                          'CC Train Band': [], 'CC Test Band': [],
                          'MSE Train Band': [], 'MSE CV Band': [], 'MSE Test Band': [], 
                          'R2 Train Band': [], 'R2 CV Band': [], 'R2 Test Band': [],
                          'EV Train Band': [], 'EV CV Band': [], 'EV Test Band': [], 
                          'CC Train PLSR': [], 'CC Test PLSR': [],
                          'MSE Train PLSR': [], 'MSE CV PLSR': [], 'MSE Test PLSR': [], 
                          'R2 Train PLSR': [], 'R2 CV PLSR': [], 'R2 Test PLSR': [],
                          'EV Train PLSR': [], 'EV CV PLSR': [], 'EV Test PLSR': []
                          })

    trainFName =args.data + 'Train_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    testFName = args.data + 'Test_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    
    
    testDF = pd.read_csv(testFName, sep = ';')
    trainDF = pd.read_csv(trainFName, sep = ';')
    testDF.rename(columns={str(compuesto):'Class'}, inplace=True)
    trainDF.rename(columns={str(compuesto):'Class'}, inplace=True)
    testDF = testDF.drop(['ID', 'Et_'+ str(compuesto),'sc_' + str(compuesto)], axis=1)
    trainDF = trainDF.drop(['ID', 'Et_'+ str(compuesto),'sc_' + str(compuesto)], axis=1)

    if 'Sand' in trainDF.columns:
        trainDF = trainDF.drop('Sand', axis=1)
    if 'Silt' in trainDF.columns:
        trainDF = trainDF.drop('Silt', axis=1)
    if 'Clay' in trainDF.columns:
        trainDF = trainDF.drop('Clay', axis=1)

    if 'Sand' in testDF.columns:
        testDF = testDF.drop('Sand', axis=1)
    if 'Silt' in testDF.columns:
        testDF = testDF.drop('Silt', axis=1)
    if 'Clay' in testDF.columns:
        testDF = testDF.drop('Clay', axis=1)

    crrntPropBands = bestBandMetric.copy()
    crrntPropBands = crrntPropBands[crrntPropBands.property == compuesto]
    crrntPropBands = crrntPropBands.reset_index(drop = True)


    ###########################################################################    
    y_train = trainDF.Class
    X_train = trainDF.drop('Class', axis=1)
    
    y_test = testDF.Class
    X_test = testDF.drop('Class', axis=1)
    ###########################################################################
    
    
    filterBand = crrntPropBands['BestBand'][0]
    testBandDF = testDF.copy()
    trainBandDF = trainDF.copy()
    allBands = list(trainDF.columns)
    allBands.remove(filterBand)
    allBands.remove('Class')
    testBandDF = testBandDF.drop(allBands, axis=1)
    trainBandDF = trainBandDF.drop(allBands, axis=1)
    
    ##########################################################################
    y_train_band = trainBandDF.Class
    X_train_band = trainBandDF.drop('Class', axis=1)
    
    y_test_band = testBandDF.Class
    X_test_band  = testBandDF.drop('Class', axis=1)
    ##########################################################################
    
    ##########################################################################
    y_train_plsr = y_train.copy()
    X_train_plsr = X_train.copy()
    
    y_test_plsr = y_test.copy()
    X_test_plsr = X_test.copy()
    
    model_band = LinearRegression()

    for model_text in av_models:
        if model_text == 'LR':
            model = LinearRegression()
            #model_band = LinearRegression()
            
        if model_text == 'SVR':
            model = SVR(kernel='linear')
            #model_band = SVR(kernel='linear')

        if model_text == 'LASSO':
            model = LassoCV(cv=5, random_state=0, 
                            max_iter = 30000, 
                            verbose = 10,
                            n_jobs = -1)
            #model_band = LassoCV(cv=5, random_state=0, 
                            #max_iter = 30000, 
                            #verbose = 10,
                            #n_jobs = -1)
        if model_text == 'Cubist':
            model = Cubist(composite = True,
                           n_committees = 15,
               #neighbors = 2,
               #cv = 10,
                           verbose = 1)
           # model_band = Cubist(composite = True,
                                #n_committees = 15,
               #neighbors = 2,
               #cv = 10,
                               # verbose = 1)

        regressor = model.fit(X_train, y_train)
    
        scoring = {'mean_squared_error':make_scorer(mean_squared_error),
                   'r2':'r2',
                   'explained_variance':'explained_variance'}
        

        ######################################################################
        regressor = model.fit(X_train, y_train)
        dump(regressor, 'models/model-' + str(compuesto) + "-allFeatures-" + str(model_text) + ".joblib")
        
        scores =cross_validate(model,X_train, y_train, scoring = scoring) 
        
        y_prd_train = regressor.predict(X_train)
        y_prd_test = regressor.predict(X_test)

        ######################################################################
    
        ######################################################################        
        regressor_band = model_band.fit(X_train_band, y_train_band)
        dump(regressor, 'models/model-' + str(compuesto) + "-bestBand-" + str(model_text) + ".joblib")
        scores_band =cross_validate(model_band,X_train_band, y_train_band,scoring = scoring) 
                    
        y_prd_train_band = regressor_band.predict(X_train_band)
        y_prd_test_band = regressor_band.predict(X_test_band)
        ######################################################################

        cc = np.corrcoef(y_train, y_prd_train)
        cc = cc[0,1]
        cc_train = cc.round(3)

        cc = np.corrcoef(y_test, y_prd_test)
        cc = cc[0,1]
        cc_test = cc.round(3)

        cc = np.corrcoef(y_train_band, y_prd_train_band)
        cc = cc[0,1]
        cc_train_band = cc.round(3)

        cc = np.corrcoef(y_test_band, y_prd_test_band)
        cc = cc[0,1]
        cc_test_band = cc.round(3)

            
        
        for nc in range(2,7):    
            model_plsr =  PLSRegression(n_components=nc)
            if nc == 6:
                dump(regressor, 'models/model-' + str(compuesto) + "-PLSR6" + ".joblib")
            regressor_plsr = model_plsr.fit(X_train_plsr, y_train_plsr)
            scores_plsr =cross_validate(model_plsr,X_train_plsr, y_train_plsr,scoring = scoring) 

            y_prd_train_plsr = regressor_plsr.predict(X_train_plsr)
            y_prd_test_plsr = regressor_plsr.predict(X_test_plsr)
            
            cc = np.corrcoef(y_train_plsr, y_prd_train_plsr[:,0])
            cc = cc[0,1]
            cc_train_plsr = cc.round(3)
            
            cc = np.corrcoef(y_test_plsr, y_prd_test_plsr[:,0])
            cc = cc[0,1]
            cc_test_plsr = cc.round(3)
    
        ########################## METRICS ###################################        
            crrntPropMetrics = [str(compuesto),#Property
                                str(model_text),#Regression Model
                                str(list(X_train_band.columns)[0]),#Best Band
                                nc,#PLSR N Components
                                cc_train,#Corr Coef train
                                cc_test,#Corr Coef test
                                mean_squared_error(y_train, y_prd_train).round(3),#MSE train
                                np.mean(scores['test_mean_squared_error']).round(3),#MSE CV
                                mean_squared_error(y_test, y_prd_test).round(3),#MSE test
                                r2_score(y_train, y_prd_train).round(3),#R2 train
                                np.mean(scores['test_r2']).round(3),#R2 CV
                                r2_score(y_test, y_prd_test).round(3),#R2 test
                                explained_variance_score(y_train, y_prd_train).round(3),#Exp Var train
                                np.mean(scores['test_explained_variance']).round(3),#Exp Var CV
                                explained_variance_score(y_test_band, y_prd_test).round(3),#Exp Var test
                                cc_train_band,#Corr Coef train band
                                cc_test_band,#Corr Coef test band
                                mean_squared_error(y_train_band, y_prd_train_band).round(3),#MSE train
                                np.mean(scores_band['test_mean_squared_error']).round(3),#MSE CV
                                mean_squared_error(y_test_band, y_prd_test_band).round(3),#MSE test
                                r2_score(y_train_band, y_prd_train_band).round(3),#R2 train
                                np.mean(scores_band['test_r2']).round(3),#R2 CV
                                r2_score(y_test_band, y_prd_test_band).round(3),#R2 test
                                explained_variance_score(y_train_band, y_prd_train_band).round(3),#Exp Var train
                                np.mean(scores_band['test_explained_variance']).round(3),#Exp Var CV
                                explained_variance_score(y_test_band, y_prd_test_band).round(3),#Exp Var test
                                cc_train_plsr,#Corr Coef train PLSR
                                cc_test_plsr,#Corr Coef test PLSR
                                mean_squared_error(y_train_plsr, y_prd_train_plsr).round(3),#MSE train
                                np.mean(scores_plsr['test_mean_squared_error']).round(3),#MSE CV
                                mean_squared_error(y_test_plsr, y_prd_test_plsr).round(3),#MSE test
                                r2_score(y_train_plsr, y_prd_train_plsr).round(3),#R2 train
                                np.mean(scores_plsr['test_r2']).round(3),#R2 CV
                                r2_score(y_test_plsr, y_prd_test_plsr).round(3),#R2 test
                                explained_variance_score(y_train_plsr, y_prd_train_plsr).round(3),#Exp Var train
                                np.mean(scores_plsr['test_explained_variance']).round(3),#Exp Var CV
                                explained_variance_score(y_test_plsr, y_prd_test_plsr).round(3),#Exp Var test
                                ]
            print(crrntPropMetrics)
            metricsDF.loc[metricsDF.shape[0] + 1] = crrntPropMetrics
            plotName = 'Regression performance on ' + str(compuesto) + ' for test dataset with ' + str(model_text) + ', PLSR ' + str(nc) + ' components' 
            regPlotBest(compuesto, y_test, y_prd_test, y_test_band, y_prd_test_band, y_test_plsr, y_prd_test_plsr, model_text, plotName, nc, "results/", save = True)

            plotName = 'Regression performance on ' + str(compuesto) + ' for train dataset with ' + str(model_text) + ', PLSR ' + str(nc) + ' components'
            regPlotBest(compuesto, y_train, y_prd_train , y_train_band, y_prd_train_band, y_train_plsr, y_prd_train_plsr, model_text, plotName, nc, "results/", save = True)
  
    metricsDF.to_csv('results/' + str(compuesto) + '-regressor-metrics.csv', index=False, header=True, sep = ';', decimal = '.')


