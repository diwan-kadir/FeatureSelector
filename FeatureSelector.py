#!/usr/bin/env python

'''
    An Implementation of feature selection algorithm.
'''


# Importing Libraries
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso

import statsmodels.api as sm

# Meta informations.
__version__ = '1.0.3'
__authors__ = ['Kadir Diwan','Rajeev Bandi','Dhruv Manoj']
__authors_email__ = ['diwan.kadir18@siesgst.ac.in','bandi.rajeev18@siesgst.ac.in','dhruvmanoj99@gmail.com']

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

def ForwardSelector(data, target, significance_level=0.05):
    '''
    Implementation of Forward selection.
    
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
        Dependent Variable after slicing attributes using forward selection.
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

def ChiSquareSelector(X,y,filter_percent = 0.8):
    '''
    Implementation of Chi Sqaure Feature selection.    
    
    Parameters
    ----------
    X : DataFrame
        Independent Variable.
    y : DataFrame
        Dependent Variable.
    filter_percent : Float, optional
        Percentage of attributes to be selected from X.
        The default is 0.8.

    Returns
    -------
    X : DataFrame
        Dependent Variable after slicing after filtering using chi-square.
    '''

    assert filter_percent >= 0.2 , 'Filtering Percentage should be atleast 0.2 or 20% for optimum results'
    
    d = int(filter_percent * X.shape[1])
    return indexes(X,pd.DataFrame(SelectKBest(score_func=chi2 , k=d).fit(X,y).get_support()))

def RandomForestSelector(X,y):
    '''
    Implementation of Random Forest Feature selection.    
    
    Parameters
    ----------
    X : DataFrame
        Independent Variable.
    y : DataFrame
        Dependent Variable.
   
    Returns
    -------
    X : DataFrame
        Dependent Variable after slicing after filtering using Random Forest feature selector.
    '''
    return indexes(X, pd.DataFrame(SelectFromModel(RandomForestClassifier(random_state=0)).fit(X, y).get_support()))

def LassoSelector(X,y,alpha=0.05):
    '''
    Implementation of Lasso Feature selection.
  
    Parameters
    ----------
    X : DataFrame
        Independent Variable.
    y : DataFrame
        Dependent Variable.
    alpha : Float, optional
        Constant that multiplies the L1 term. The default is 0.05.
    Returns
    -------
    X : DataFrame
        Dependent Variable after slicing after filtering using Lasso feature selector.
    '''
    assert alpha !=0 , 'Alpha cannot be zero'
    
    return indexes(X, pd.DataFrame(SelectFromModel(Lasso(alpha,random_state=0)).fit(X, y).get_support()))

def Pipelined(X,y,alpha=0.05,filter_percent=0.8):
    '''
    Feature Selection Algorithms in a Pipelined Manner.
    
    
    Filtering Stages
    ----------------    
    ChiSquare => Randome Forest => Forward Selection => Lasso
    
    
    Parameters
    ----------
    X : DataFrame
        Independent Variable.
    y : DataFrame
        Dependent Variable.
    alpha : Float, optional
        Constant that multiplies the L1 term.Used for Lasso The default is 0.05.
    filter_percent : Float, optional
        Percentage of attributes to be selected from X.Used in Chi-Square
        The default is 0.8.

    Returns
    -------
    X : DataFrame
        Dependent Variable after slicing after filtering using Pipelined feature selector.
    '''
    return LassoSelector(ForwardSelector(RandomForestSelector(ChiSquareSelector(X,y,filter_percent),y),y),y,alpha)

