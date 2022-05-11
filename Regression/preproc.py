# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from findiff import FinDiff
from sklearn.model_selection import train_test_split
from pickle import dump
import argparse
from joblib import dump, load

parser = argparse.ArgumentParser(description='Pre processing of NIRS data.')
parser.add_argument('-d', '--data', type=str,
                    help='Full NIRS data path, must be CSV formatted, and ; separated')
parser.add_argument('-t', '--threshold', type=str,
                    help='full thresholds table data set, must be CSV formatted, and ; separated')
parser.add_argument('-p', '--properties', type=str,
                    help='Space separated list of properties in "quotes", must be contained in properties data set header')
parser.add_argument('-s', '--step', type=str,
                    help='Space separated steps in NIRS spectrum (float or INT), begin step end')



args = parser.parse_args()

def argTolist(argStr, num = False):
    l = []
    for t in argStr.split():
        if num:
            l.append(float(t))
        else:
            l.append(str(t))
    return l

print(args.data)
print(args.threshold)
print(args.properties)
print(argTolist(args.properties))
print(args.step)
print(argTolist(args.step, True))


#df = pd.read_csv('data/soil_cane_vis-NIR.csv' ,sep = ';', decimal = '.')
df = pd.read_csv(args.data ,sep = ';', decimal = '.')
#umbrales = pd.read_csv('data/properties_thresholds.csv' ,sep = ';', decimal = '.')
umbrales = pd.read_csv(args.threshold ,sep = ';', decimal = '.')
#otrosCompuestos = ['pH_3Cat','pH','OM', 'Ca', 'Mg', 'K', 'Na']
otrosCompuestos = argTolist(args.properties)

steps = argTolist(args.step, True)

e = []
x = []
for i in np.arange(steps[0],steps[2],steps[1]):
    if i-int(i) == 0:
        e.append(str(int(i)))
        x.append(int(i))
    else:
        e.append(str(i))
        x.append(i)

#scalers = ['minmax', 'standard', 'powertrans','quantile']
#scalers = ['minmax','standard']
scalers = ['minmax']



for avScaler in scalers:

    espectro = df.copy()
    espectro=espectro[e]
    #espectro = espectro.drop(otrosCompuestos, axis=1)
    #espectro = espectro.drop(['Sand','Clay','Silt'], axis=1)
    
    Sand = df.Sand
    Clay = df.Clay
    Silt = df.Silt
    
    DFCompuestos = df.copy()
    DFCompuestos = DFCompuestos[otrosCompuestos]
                  
    aespectro = np.asarray(espectro)    
    X = list(espectro.columns)
    dx = float(X[1]) - float(X[0])
    d_dx = FinDiff(0, dx, 1, acc = 10)
    d2_dx = FinDiff(0, dx, 2, acc = 10)
    
    AD1 = None
    AD2 = None
    
    for i in range(0, len(espectro)):
        d1 = d_dx(aespectro[i])
        d1 = d1.reshape(1,-1)
        print(d1)
        d2 = d2_dx(aespectro[i])
        d2 = d2.reshape(1,-1)
        fftt = np.fft.fft(aespectro[i])
        fftt = fftt.reshape(1,-1)
        print(d2)
        if i == 0:
            ad1 = d1
            ad2 = d2
            afft = fftt
        else:
            ad1 = np.append(ad1, d1, axis=0)
            ad2 = np.append(ad2, d2, axis=0)
            afft = np.append(afft, fftt, axis=0)
    
    if avScaler == 'minmax':
        scalerEspectro = MinMaxScaler()
        scalerd1 = MinMaxScaler()
        scalerd2 = MinMaxScaler()
        sc_l = MinMaxScaler()
        
    if avScaler == 'standard':
        scalerEspectro = StandardScaler()
        scalerd1 = StandardScaler()
        scalerd2 = StandardScaler()
        sc_l = StandardScaler()
        
    if avScaler == 'powertrans':
        scalerEspectro = PowerTransformer()
        scalerd1 = PowerTransformer()
        scalerd2 = PowerTransformer()
        sc_l = PowerTransformer()

    if avScaler == 'quantile':
        scalerEspectro = QuantileTransformer()
        scalerd1 = QuantileTransformer()
        scalerd2 = QuantileTransformer()
        sc_l = QuantileTransformer()
        
    #x = np.arange(400,2492,8.5)
    x = np.arange(steps[0],steps[2],steps[1])
    scalerEspectro.fit(aespectro)
    scalerd1.fit(ad1)
    scalerd2.fit(ad2)
    
    dump(scalerEspectro, 'scalers/spectrum-' + str(avScaler) + ".joblib")
    dump(scalerd1, 'scalers/d1-' + str(avScaler) + ".joblib")
    dump(scalerd1, 'scalers/d2-' + str(avScaler) + ".joblib")
    
    scEspectro = scalerEspectro.transform(aespectro)
    scad1 = scalerd1.transform(ad1)
    scad2 = scalerd2.transform(ad2)
    
    
    absfft = np.absolute(afft)/int(afft.shape[1])
    np.savetxt("scalers/fft.csv", absfft, delimiter=";")
    allData = np.concatenate((scEspectro, scad1, scad2, absfft), axis = 1)
    
    allDataColumns = []
    
    for j in X:
        allDataColumns.append('scE_' + str(j))
        
    for j in X:
        allDataColumns.append('d1_' + str(j))
        
    for j in X:
        allDataColumns.append('d2_' + str(j))
    
    for j in X:
        allDataColumns.append('fft_' + str(j))
 
    allDFNC = pd.DataFrame(data = allData, columns = allDataColumns)


    for i in range(0,len(otrosCompuestos)):  
        allDF = allDFNC.copy()
        compuesto = otrosCompuestos[i]
        labels = list(DFCompuestos.iloc[:,i])
        
        
        comp_et = []
        umbFilt = umbrales[umbrales.Compuesto == compuesto]
        for comp_i in labels:   
            for index, row in umbFilt.iterrows():
                if comp_i > row['limInf'] and comp_i <= row['limSup']:
                    comp_et.append(row['Etiqueta'])
        
        labels = np.array(labels).reshape(-1,1)
        sc_l.fit(labels)
        
        dump(sc_l, 'scalers/' + str(compuesto) + "-" + str(avScaler) + ".joblib")
        
        scLabels = sc_l.transform(labels)
        scLabels = scLabels.flatten()
        labels = labels.flatten()
        
        ids = list(range(0,len(labels)))
        
        allDF.insert (0, 'Silt', Silt)
        allDF.insert (0, 'Clay', Clay)
        allDF.insert (0, 'Sand', Sand)
        
        allDF.insert (0, 'sc_' + str(compuesto), scLabels)
        allDF.insert (0, str(compuesto), labels)
        allDF.insert (0, 'Et_' + str(compuesto), comp_et)
        allDF.insert (0, 'ID', ids)
        
        allDF.to_csv('data/All_'+ avScaler + '_d1d2_fft_feature_' + str(compuesto) + '.csv', index=False, header=True, sep = ';', decimal = '.') 
        
        train, test = train_test_split(allDF, test_size=.3, random_state=(32))
        
        train.to_csv('data/Train_'+ avScaler + '_d1d2_fft_feature_' + str(compuesto) + '.csv', index=False, header=True, sep = ';', decimal = '.') 
        test.to_csv('data/Test_'+ avScaler + '_d1d2_fft_feature_' + str(compuesto) + '.csv', index=False, header=True, sep = ';', decimal = '.') 
        