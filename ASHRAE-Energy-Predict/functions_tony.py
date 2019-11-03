import numpy as np
import pandas as pd
import os, warnings, math
import time
import datetime

## Function to reduce memory
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# adapted from @kyakovlev

def F_mem_reducer(df):
    for col in df.columns:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
# adapted from @kyakovlev

def reduce_mem_usage(df, verbose=True):
    if verbose: 
        start_mem = df.memory_usage().sum() / 1024**2
        F_mem_reducer(df)
        end_mem = df.memory_usage().sum() / 1024**2
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    else:
        F_mem_reducer(df)
        

## Count unique and missing values
## :df pandas dataframe already sliced to the columns of interest             # type: pd.DataFrame()

def count_uniq_miss(df):
    for col in df.columns:
        u = df[col].nunique()
        m = df[col].isnull().sum()
        n = df.shape[0]
        print('Column `{0}` has {1:2d} unique values and {2:2d} values missing ({3:.0f}% of total).'.format(col,u,m,m/n))
        
