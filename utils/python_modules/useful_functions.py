import pandas
import numpy
import matplotlib
import seaborn
import IPython
import scipy

import notebook.notebookapp
import urllib
import json
import os
import ipykernel
import shutil

import pickle
import os

def plot_correlation_matrix_heat_map(df,label,qty_fields=10):
    df = pandas.concat([df[label],df.drop(label,axis=1)],axis=1)
    correlation_matrix = df.corr()
    index = correlation_matrix.sort_values(label, ascending=False).index
    correlation_matrix = correlation_matrix[index].sort_values(label,ascending=False)

    fig,ax = matplotlib.pyplot.subplots()
    fig.set_size_inches((10,10))
    seaborn.heatmap(correlation_matrix.iloc[:qty_fields,:qty_fields],annot=True,fmt='.2f',ax=ax)
    
    # Code added due to bug in matplotlib 3.1.1
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + .5, top - .5)

    return(fig,ax)

def plot_log_hist(s,bin_factor=1,min_exp=None):
    """Plot log histogram. Bin_factor increases the number of bins
    to use for the histogram, min_exp sets the minimum exponent to use
    in creating the bins. If not supplied, the min_exp is calculated 
    automatically."""
    # Determine the max x value for the plot
    x_max = numpy.ceil(numpy.log10(max(s)))

    # If min_exp not specified, calculate an appropriate min_exp by choosing
    # a value where 90% of the values are above that min_exp.
    if min_exp == None:
        for i in range(10):
            n_betw = s[s!=0].between(0,10**-i).sum()
            if not (n_betw / s[s!=0].shape[0]) > .1:
                min_exp = -i
                break

    # Clip values lower than min_exp to the min_exp so they'll show up in the plot.
    s = s.clip(lower=10**min_exp)

    # Create bins to use in plot
    bins = numpy.logspace(min_exp, x_max, (x_max + 1)*bin_factor)
    
    # Plot histogram
    fig,ax = matplotlib.pyplot.subplots()
    s.hist(bins=bins,ax=ax)
    ax.set_xscale('log')
    return(fig,ax)

def get_null_counts(df):
    null_df = pandas.DataFrame(df.isnull().sum(),columns=['null_count'])
    null_df['null_fraction'] = null_df['null_count'] / df.shape[0]
    null_df = null_df.sort_values('null_count',ascending=False)
    return null_df

def get_zero_counts(df):
    zero_counts = pandas.DataFrame((df==0).sum(), columns=['zero_count'])
    zero_counts['zero_fraction'] = zero_counts['zero_count'] / df.shape[0]
    zero_counts = zero_counts.sort_values('zero_count',ascending=False)
    return(zero_counts)

def get_null_and_zero_counts(df):
    df_nz = pandas.DataFrame(df.isnull().sum(),columns=['null_count'])
    df_nz['null_fraction'] = df_nz['null_count'] / df.shape[0]
    
    df_nz['zero_counts'] = (df==0).sum()
    df_nz['zero_fraction'] = df_nz['zero_counts'] / df.shape[0]
    
    df_nz['zero_or_null_counts'] = ((df == 0) | (df.isnull())).sum()
    df_nz['zero_or_null_fraction'] = df_nz['zero_or_null_counts'] / df.shape[0]
    
    df_nz = df_nz.sort_values('zero_or_null_counts',ascending=False)
    
    return(df_nz)

def top_value_counts(df, n=5, only_categories = True, cols_to_include = None, cols_to_exclude = None):
    """ Function to generate summary information for string or categorical
    data in dataframes"""

    if cols_to_include:
        df = df[cols_to_include]
    if cols_to_exclude:
        df = df[df.columns[~df.columns.isin(cols_to_exclude)]]
        
    if 'float' in list(df.dtypes):
        print("Error, column(s) with float dtype included")
        print('The following columns will be excluded',list(df.select_dtypes(include='float64').columns))
    
    df = df.select_dtypes(exclude='float64')
    
    if only_categories:
        df = df.select_dtypes(include=['O','category'])
        
    cols = df.columns
    df_value_counts = pandas.DataFrame()
    i_name = -1
    for col in cols:
        i_name += 1
        counts = df[col].value_counts(dropna=False)[:n]
        top_n_names = list(counts.index)
        top_n = list(counts)
        if len(top_n) < n+1:
            for i in range(n-len(top_n)):
                top_n.append('-')
                top_n_names.append('-')
        top_n_names.insert(0,'n_unique')
        top_n.insert(0,df[col].nunique())
        df_value_counts[col] = top_n_names
        df_value_counts[i_name] = top_n
    new_index = pandas.MultiIndex.from_product([df_value_counts.columns[range(0,len(df_value_counts.columns),2)],('Cat','Freq')],names = ['field','info'])
    df_value_counts.columns = new_index
    return(df_value_counts)


