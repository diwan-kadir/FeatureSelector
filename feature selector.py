#!/usr/bin/env python

'''
    An Implementation of feature selection algorithm.
'''

# Importing Libraries
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression

import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Meta informations.
__version__ = '1.0.3'
__author__ = 'Kadir Diwan'
__author_email__ = 'diwan.kadir18@siesgst.ac.in'

def read_path(path,m=0,n=-1):
    '''
    Read path of csv file and return the Dataframe of X and y (Independent and Dependent Variables).
    
    Note : 0 indexing considered.
    
    Parameters
    ----------
    path : String
        absolute path of the csv file.
    m : Integer, optional
        Starting column number (index included). The default is 0.
    n : Integer, optional
        Ending column number (index not included). The default is -1.

    Returns
    -------
    X : DataFrame
        Independent Variable.
    y : DataFrame
        Dependent Variable.


    Example
    -------
    Input :
        (['a','b','c','d','e','f','Output'])
        for m = 1 , n = 5
    Output :
        X => ['b','c','d','e']
        y => ['Output']
        
    '''

    df = pd.read_csv(path)
    assert n < df.shape[1], 'n out of bounds \n\t\t\t i.e n exceeded number of columns'
        
    return df.iloc[:,m:n],df.iloc[:,-1]

def indexes(X,ar):
    '''
    
    Slices Dataframe Given as per given indexed Boolean Dataframe.
    
    Parameters
    ----------
    X : DataFrame
        Dependent Variables with column names
    ar : DataFrame of Boolean
        Array of Boolean indicating Attributes to be selected or sliced.

    Returns
    -------
    X : Datframe.
        Dependent Variable after slicing of False indexes.
    
    Example    
    -------
    Input : 
        X = ['A','B','C','D','E','F','G','H']
        ar = [True, False,True, False,True, False,True, False]
        
    Output :
        ['A','C','E','G']

    '''
    index_array = []
    c = 0
    for i in list(ar[0]):
        if i == True:
            index_array.append(c)
        c+=1
    # print(index_array)
    return X.iloc[:,index_array]

def forward_selection(data, target, significance_level=0.05):
    '''
    Returns Sliced DataFrame of Dependent Variables after filtering out features through Forward Selection.
    
    Parameters
    ----------
    data : DataFrame
        Independent Variable.
    target : DataFrame
        Dependent Variable.
    significance_level : Integer, optional
        The default is 0.05.

    Returns
    -------
    X : DataFrame
        Dependent Variable after slicing of False indexes.

    '''
    data_replica = data
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features,dtype=float)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    data = list(data)
    for i in range(len(data)):
        if data[i] in best_features:
            data[i] = True
        else :
            data[i] = False
    return indexes(data_replica, pd.DataFrame(data))


