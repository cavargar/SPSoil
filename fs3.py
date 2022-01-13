# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:49:03 2020

@author: ddelgadillo
"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, SelectPercentile, chi2, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import f_regression


otrosCompuestos = ['pH','OM', 'Ca', 'Mg', 'K','Na']


#compuesto = otrosCompuestos[5]



for compuesto in otrosCompuestos:
    trainFName = 'Train_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    testFName = 'Test_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    
    
    testDF = pd.read_csv(testFName, sep = ';')
    trainDF = pd.read_csv(trainFName, sep = ';')
    testDF.rename(columns={'sc_' + str(compuesto):'Class'}, inplace=True)
    trainDF.rename(columns={'sc_' + str(compuesto):'Class'}, inplace=True)
    testDF.rename(columns={'Et_' + str(compuesto):'Label'}, inplace=True)
    trainDF.rename(columns={'Et_' + str(compuesto):'Label'}, inplace=True)
    testDF = testDF.drop(['ID', str(compuesto)], axis=1)
    trainDF = trainDF.drop(['ID', str(compuesto)], axis=1)
    
    
    y_train_num = trainDF.Class
    y_train_cat = trainDF.Label
    x_train = trainDF.drop(['Class', 'Label'], axis=1)
    
    y_test_num = testDF.Class
    y_test_cat = testDF.Label
    x_test = testDF.drop(['Class', 'Label'], axis=1)
    
    #------ Feature Selection Classification -------
    
    features = x_train.columns
    
    if compuesto == 'pH':
        featureDF = pd.DataFrame({'feature': features})
        varianceDF = pd.DataFrame({'feature': features})
        univariateDF = pd.DataFrame({'feature': features})
        FScoreDF = pd.DataFrame({'feature': features})
        
    
    
    #  Variance = p*(1-p) p % zeros by feature
    v = 0.1
    p = 0.8
    
    #sel = VarianceThreshold(threshold=(p * (1 - p)))
    sel = VarianceThreshold(threshold=(v))
    sel.fit_transform(x_train)
    #x_n = sel.fit_transform(x_train)
    featureDF['Variance'] = sel.variances_
    varianceDF[str(compuesto)] = sel.variances_
    featureDF['selected_by_Variance'] = sel.get_support()
    
    
    # Univariate 
    #k Best
    selk = SelectKBest(mutual_info_classif,k=247)
    selk.fit(x_train, y_train_cat)
    #x_n = selk.fit_transform(x_train,y_train_cat)
    featureDF['MID_kb_Score'] = selk.scores_
    featureDF['selected_by_MID_kb'] = selk.get_support()
    
    
    # Percentile
    selp = SelectPercentile(mutual_info_classif,percentile=10)
    selp.fit(x_train, y_train_cat)
    x_n = selp.fit_transform(x_train,y_train_cat)
    featureDF['MID_per_Score'] = selp.scores_
    featureDF['selected_by_MID_perce'] = selp.get_support()
    
    
    #------ Feature Selection Regression -------
    
    
    # Univariate 
    #k Best
    selkr = SelectKBest(mutual_info_regression,k=100)
    selkr.fit(x_train, y_train_num)
    #x_n = selk.fit_transform(x_train,y_train_cat)
    featureDF['rMID_kb_Score'] = selkr.scores_
    featureDF['selected_by_rMID_kb'] = selkr.get_support()
    
    
    # Percentile
    selpr = SelectPercentile(mutual_info_regression,percentile=10)
    selpr.fit(x_train, y_train_num)
    #x_n = selpr.fit_transform(x_train,y_train_num)
    featureDF['rMID_per_Score'] = selpr.scores_
    featureDF['selected_by_rMID_perce'] = selpr.get_support()
    
    
    selkf_reg = SelectKBest(f_regression,k=100)
    selkf_reg.fit(x_train, y_train_num)
    #featureDF[str(cols[0])] = selkf_reg.scores_
    #featureDF[str(cols[1])] = selkf_reg.pvalues_
    #featureDF[str(cols[2])] = selkf_reg.get_support()
    FScoreDF[str(compuesto)] = selkf_reg.scores_
    featureDF[str(compuesto) + '_FScore'] = selkf_reg.scores_
    #Lasso Importace
    clf = LassoCV(cv=5, random_state=0,max_iter = 10000, verbose = 10, n_jobs = -1).fit(x_train, y_train_num)
    featureDF['Lasso_importance'] = np.abs(clf.coef_)
#clf.get_support()

FScoreDF = FScoreDF.set_index('feature')
from sklearn.preprocessing import MinMaxScaler
cols = FScoreDF.columns
rws = FScoreDF.index
scFScore = MinMaxScaler(feature_range = (0,100)).fit(FScoreDF).transform(FScoreDF)

tFScoreDF = np.transpose(scFScore)

cols = FScoreDF.columns
rws = FScoreDF.index



import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
        z = tFScoreDF,
        x = rws,
        y = cols,
        colorscale='Viridis'))
fig.write_html("FScore.html",)

varianceDF2 = varianceDF.copy   
varianceDF2 = varianceDF2.iloc[3:,:] 
varianceDF2 = varianceDF2.set_index('feature')

cols = varianceDF2.columns
rws = varianceDF2.index
scvariance = MinMaxScaler(feature_range = (0,100)).fit(varianceDF2).transform(varianceDF2)

tscvariance = np.transpose(scvariance)




fig = go.Figure(data=go.Heatmap(
        z = tscvariance,
        x = rws,
        y = cols,
        colorscale='Viridis'))
fig.write_html("variance.html",)    
    
