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
    
    Note : 0 indexing considered
    
    Arguments:
        path -- absolute path of the csv file.
        m -- Starting column number (index included) 
        n  -- Ending column number (index not included)
        
        eg. 
        (['a','b','c','d','e','f'])
        for m = 1 , n = 5
        
        X => ['b','c','d','e']
   
    '''
    
    df = pd.read_csv(path)
    assert n < df.shape[1], 'n out of bounds \n\t\t\t i.e n exceeded number of columns'
        
    return df.iloc[:,m:n],df.iloc[:,-1]


def indexes(X,ar):
    index_array = []
    c = 0
    for i in list(ar[0]):
        if i == True:
            index_array.append(c)
        c+=1
    # print(index_array)
    return X.iloc[:,index_array]

X,y = read_path('D:\Internship\dataset.csv')
columns = pd.DataFrame(list(X.columns))

forward_selection = pd.DataFrame(forward_selection(X, y))
X0 = indexes(X,forward_selection)

d = int(X0.shape[1]*0.80)

chi_square = pd.DataFrame(SelectKBest(score_func=chi2 , k=d).fit(X0,y).get_support())
X1 = indexes(X0,chi_square)
