# -*- coding: utf-8 -*-
"""
Created on Mon May 30 08:41:38 2016

@author: atreo
"""

import numpy as np
import pandas as pd

       
def mean_abs(array):    
    return np.mean(np.abs(array[:]))
    
    
def weight_mean(net, key):
    w_mean = mean_abs(net.params[key][0].data)    
    return w_mean
    

def bias_mean(net, key):
    b_mean = mean_abs(net.params[key][1].data)    
    return b_mean
    

def print_means(net):
    """print mean of the absolute value of 2 parameters per layer
       (weight and bias)
       prints a warning when a layer
    """
    for p in net.params:
        try:        
            print('<weight> of', p, ':', weight_mean(net, p))
        except IndexError:
            print('Layer', p, ' has no weight')
        try:            
            print('<bias> of', p, ':', bias_mean(net, p))
        except IndexError:
            print('Layer', p, ' has no bias')


def layer_list(net):
    """creates a list with all layernames
    """
    layers = []    
    for p in net.params:
        layers.append(str(p))
    return layers


def gen_pandas_struct(net):
    """creates a list with 2 arrays of layernames and parameterkeys
       for pandas datastructure
    """
    wblist = ['NumIters']
    nlist = ['']
    for p in net.params:
        wblist.append('weight')
        wblist.append('bias') 
        nlist.append(p)
        nlist.append(p)   
    pdstruct = [np.array(nlist), np.array(wblist)]
    return pdstruct
    
    
def get_mean_df(net, iteration):
    """extracts up to N = 2 parameters per layer (weight and bias)
       and takes the mean of the absolute values
       
       result is stored in a multi-indexed dataframe
    """
    N = 2
    data = np.ndarray((1,N*len(net.params)+1))
    data[0,0] = int(iteration)
    i = 0
    for p in net.params:
        try:
            w_mean = weight_mean(net, p)
        except IndexError:
#            print('Layer', p, ' has no weight')
            w_mean = np.NaN
        try:
            b_mean = bias_mean(net, p)
        except IndexError:
#            print('Layer', p, ' has no bias')
            b_mean = np.NaN
        data[0, 2*i+1] = w_mean
        data[0, 2*i+2] = b_mean
        i += 1   
    df = pd.DataFrame(data, columns=gen_pandas_struct(net))
    return df