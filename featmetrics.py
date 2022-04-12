# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import f_regression
import argparse





parser = argparse.ArgumentParser(description='Pre processing of NIRS data.')
parser.add_argument('-d', '--data', type=str,
                    help='Full NIRS data path, must be CSV formatted, and ; separated')
parser.add_argument('-p', '--properties', type=str,
                    help='Space separated list of properties in "quotes", must be contained in properties data set header')
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
otrosCompuestos = argTolist(args.properties)

#otrosCompuestos = ['pH','OM', 'Ca', 'Mg', 'K','Na']






for compuesto in otrosCompuestos:
    trainFName = args.data + 'Train_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    testFName = args.data + 'Test_minmax_d1d2_fft_feature_' + str(compuesto) + '.csv'
    
    
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
    featureDF = pd.DataFrame({'feature': features})
    
    #  Variance = p*(1-p) p % zeros by feature
    v = 0.1
    p = 0.8
    
    sel = VarianceThreshold(threshold=(v))
    sel.fit_transform(x_train)
    featureDF['Variance'] = sel.variances_
   
    # Percentile
    selpr = SelectPercentile(mutual_info_regression,percentile=10)
    selpr.fit(x_train, y_train_num)
    featureDF['rMID_per_Score'] = selpr.scores_
    
    # FScore
    selkf_reg = SelectKBest(f_regression,k=100)
    selkf_reg.fit(x_train, y_train_num)
    featureDF['FScore'] = selkf_reg.scores_
    
    #Lasso Importace
    clf = LassoCV(cv=5, random_state=0,max_iter = 10000, verbose = 10, n_jobs = -1).fit(x_train, y_train_num)
    featureDF['Lasso_importance'] = np.abs(clf.coef_)

    #Correlation
    corr = []
    for f in features:
        #print(f)
        cc = np.corrcoef(x_train[f], y_train_num)
        corr.append(cc[0,1])
    featureDF['Correlation'] = corr
    featureDF.to_csv('feature_metrics/metrics_' + str(compuesto) + '.csv', index=False, header=True, sep = ';', decimal = '.')
    