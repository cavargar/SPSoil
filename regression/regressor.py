# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:24:37 2020

@author: ddelgadillo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
from pickle import dump
import six
import regplots as rgp
from sklearn.cross_decomposition import PLSRegression

#otrosCompuestos = ['pH','OM', 'Ca', 'Mg', 'K','Na']
otrosCompuestos = ['Ca']
#otrosCompuestos = ['K','Na']
#otrosCompuestos = ['Mg']
#otrosCompuestos = ['pH']
#av_models = ['LR']
#av_models = ['LR']
av_models = ['LR', 'SVR','LASSO']



bestBandMetric = pd.DataFrame({'property':[],
                         'metric':[],
                         'BestBand':[],
                         'Score':[]
                        })


compuesto = otrosCompuestos[0]
for compuesto in otrosCompuestos:
    propMetrics = pd.read_csv('metricas_propiedad/metricas_' + str(compuesto) + '.csv',sep = ';', decimal = '.')
    
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



compuesto = otrosCompuestos[0]
for compuesto in otrosCompuestos:
    
    
    trainFName = 'data/Train_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    testFName = 'data/Test_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    
    
    testDF = pd.read_csv(testFName, sep = ';')
    trainDF = pd.read_csv(trainFName, sep = ';')
    testDF.rename(columns={str(compuesto):'Class'}, inplace=True)
    trainDF.rename(columns={str(compuesto):'Class'}, inplace=True)
    testDF = testDF.drop(['ID', 'Et_'+ str(compuesto),'sc_' + str(compuesto), 'Sand', 'Clay', 'Silt'], axis=1)
    trainDF = trainDF.drop(['ID', 'Et_'+ str(compuesto),'sc_' + str(compuesto), 'Sand', 'Clay', 'Silt'], axis=1)
    
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
    
    
    ##########################################################################    
    #crrntCompMetrics = [str(compuesto), str(crrntPropBands['metric'][0]), str(crrntPropBands['Score'][0]), str(crrntPropBands['BestBand'][0])]
    

    #crrntCompMetricsBest = [str(compuesto), str(crrntPropBands['metric'][0]), str(crrntPropBands['Score'][0]), str(best_bands)]





    for model_text in av_models:
        if model_text == 'LR':
            model = LinearRegression()
            model_band = LinearRegression()
            
        if model_text == 'SVR':
            model = SVR(kernel='linear')
            model_band = SVR(kernel='linear')

        if model_text == 'LASSO':
            model = LassoCV(cv=5, random_state=0, 
                            max_iter = 30000, 
                            verbose = 10,
                            n_jobs = -1)
            model_band = LassoCV(cv=5, random_state=0, 
                            max_iter = 30000, 
                            verbose = 10,
                            n_jobs = -1)
    
        scoring = {'mean_squared_error':make_scorer(mean_squared_error),
                   'r2':'r2',
                   'explained_variance':'explained_variance'}
        

        ######################################################################
        regressor = model.fit(X_train, y_train)
        scores =cross_validate(model,X_train, y_train, scoring = scoring) 
        
        y_prd_train = regressor.predict(X_train)
        y_prd_test = regressor.predict(X_test)

        ######################################################################
    
        ######################################################################        
        regressor_band = model_band.fit(X_train_band, y_train_band)
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
                
            regressor_plsr = model_plsr.fit(X_train_plsr, y_train_plsr)
            scores_plsr =cross_validate(model_plsr,X_train_plsr, y_train_plsr,scoring = scoring) 

            y_prd_train_plsr = regressor_plsr.predict(X_train_plsr)
            y_prd_test_plsr = regressor_plsr.predict(X_test_plsr)
            
            cc = np.corrcoef(y_train_plsr, y_prd_train_plsr[:,0])
            cc = cc[0,1]
            cc_train_plsr = cc.round(3)
            #crrntCompMetrics.append(cc_train)
            
            cc = np.corrcoef(y_test_plsr, y_prd_test_plsr[:,0])
            cc = cc[0,1]
            cc_test_plsr = cc.round(3)
            #crrntCompMetrics.append(cc_test)    
    
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
            plotName = 'Regression performance on ' + str(compuesto) + ' for test dataset with ' + str(model_text) + ', PLSR ' + str(nc) + ' components' 
            rgp.regPlotBest3(compuesto, y_test, y_prd_test, y_test_band, y_prd_test_band, y_test_plsr, y_prd_test_plsr, model_text, plotName, nc, save = True)

            plotName = 'Regression performance on ' + str(compuesto) + ' for train dataset with ' + str(model_text) + ', PLSR ' + str(nc) + ' components'
            rgp.regPlotBest3(compuesto, y_train, y_prd_train , y_train_band, y_prd_train_band, y_train_plsr, y_prd_train_plsr, model_text, plotName, nc, save = True)
  
            metricsDF.loc[metricsDF.shape[0] + 1] = crrntPropMetrics

