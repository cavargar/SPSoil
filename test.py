# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 05:18:27 2022

@author: ddelgadillo
"""
import argparse

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
