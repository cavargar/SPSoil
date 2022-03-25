# -*- coding: utf-8 -*-
"""
Regression main script

@author: ddelgadillo
"""
import yaml
import pprint
import os


def readYaml():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
 
    return config
 
if __name__ == "__main__":
 
    # read the config yaml
    config = readYaml()
 
    # pretty print my_config
    #pprint.pprint(config)
    
    preProc = "python preproc.py -d " + str(config['data']['dataPath'][0]) + str(config['data']['dataFile'][0]) + " -t " + str(config['data']['thresholdsPath'][0]) + ' -p "' + str(config['properties'][0]) + '" -s "' + str(config['NIRSInfo']['steps'][0]) + '"' 
    print(preProc)
    os.system(preProc)
    #python .\preproc.py -d data/soil_cane_vis-NIR.csv -t data/properties_thresholds.csv -p "pH_3Cat pH OM Ca Mg K Na" -s "400 8.5 2492"
    
    regressor = 'python regression/regressor.py -p "' + str(config['properties'][0]) + '" -m "' + str(config['Regression']['models'][0]) +'" -f feature_metrics/ -d ' + str(config['data']['dataPath'][0])
    print(regressor)
    os.system(regressor)
    #python .\regression\regressor.py -p "Ca" -m "LR" -f feature_metrics\ -d data\