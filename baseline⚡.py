#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgb
import math
from tqdm import tqdm_notebook as tqdm
import warnings
import datetime as dt
warnings.simplefilter(action='ignore', category=FutureWarning)

PATH = 'data/'
#!ls ../input/ashrae-energy-prediction


# **Reduce Memory function**

# In[2]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# **RMSLE calculation** 

# In[3]:


def rmsle(y, y_pred):
    '''
    A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
    source: https://www.kaggle.com/marknagelberg/rmsle-function
    '''
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# In[4]:


# from: https://www.kaggle.com/bejeweled/ashrae-catboost-regressor
def RMSLE(y_true, y_pred, *args, **kwargs):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# **Feature Functions**

# In[5]:


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]


# # Data

# In[6]:


building_df = pd.read_csv(PATH+"building_metadata.csv")
weather_train = pd.read_csv(PATH+"weather_train.csv", parse_dates=['timestamp'])
weather_test = pd.read_csv(PATH+"weather_test.csv", parse_dates=['timestamp'])
train = pd.read_csv(PATH+"train.csv", parse_dates=['timestamp'])


# ### Prepare training and test

# In[7]:


weather = pd.concat([weather_train,weather_test],ignore_index=True)
weather_key = ['site_id', 'timestamp']

temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()

# calculate ranks of hourly temperatures within date/site_id chunks
temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')

# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)

# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
site_ids_offsets.index.name = 'site_id'

def timestamp_align(df):
    df['offset'] = df.site_id.map(site_ids_offsets)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    return df


# In[8]:


weather_train = timestamp_align(weather_train)
weather_test = timestamp_align(weather_test)


# In[9]:


add_lag_feature(weather_train, window=3)
add_lag_feature(weather_train, window=72)
add_lag_feature(weather_test, window=3)
add_lag_feature(weather_test, window=72)


# In[10]:


train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])


# In[11]:


#test = test.merge(weather_test, left_on = ["timestamp"], right_on = ["timestamp"])
#del weather_test


# # Simple FE: Timestamp

# In[12]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train["hour"] = train["timestamp"].dt.hour
train["day"] = train["timestamp"].dt.day
train["weekend"] = train["timestamp"].dt.weekday
train["month"] = train["timestamp"].dt.month
train["year"] = train["timestamp"].dt.year
print ('TRAIN: ', train.shape)
train.head(3)


# **Delete time stamp and encode ```primary_use```**

# In[13]:


train = train.drop("timestamp", axis = 1)
le = LabelEncoder()
train["primary_use"] = le.fit_transform(train["primary_use"])


# In[14]:


train.head(3)


# In[15]:



categoricals = ["building_id", "primary_use", "hour", "day", "weekend", "month", "meter", 'year']

drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature"]

feat_cols = categoricals + numericals


# In[16]:


target = np.log1p(train["meter_reading"])


# In[17]:


train = train.drop(drop_cols + ["site_id","floor_count","meter_reading"], axis = 1)
#train.fillna(-999, inplace=True)
train.head()


# In[18]:


train, NAlist = reduce_mem_usage(train);


# ## Validation

# In[19]:


# target = np.log1p(train["meter_reading"])
# raw_target = np.expm1(target)

feat_cols=feat_cols+['air_temperature_mean_lag3', 'air_temperature_max_lag3',
       'air_temperature_min_lag3', 'air_temperature_std_lag3',
       'cloud_coverage_mean_lag3', 'cloud_coverage_max_lag3',
       'cloud_coverage_min_lag3', 'cloud_coverage_std_lag3',
       'dew_temperature_mean_lag3', 'dew_temperature_max_lag3',
       'dew_temperature_min_lag3', 'dew_temperature_std_lag3',
       'precip_depth_1_hr_mean_lag3', 'precip_depth_1_hr_max_lag3',
       'precip_depth_1_hr_min_lag3', 'precip_depth_1_hr_std_lag3',
       'sea_level_pressure_mean_lag3', 'sea_level_pressure_max_lag3',
       'sea_level_pressure_min_lag3', 'sea_level_pressure_std_lag3',
       'wind_direction_mean_lag3', 'wind_direction_max_lag3',
       'wind_direction_min_lag3', 'wind_direction_std_lag3',
       'wind_speed_mean_lag3', 'wind_speed_max_lag3', 'wind_speed_min_lag3',
       'wind_speed_std_lag3', 'air_temperature_mean_lag72',
       'air_temperature_max_lag72', 'air_temperature_min_lag72',
       'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
       'cloud_coverage_max_lag72', 'cloud_coverage_min_lag72',
       'cloud_coverage_std_lag72', 'dew_temperature_mean_lag72',
       'dew_temperature_max_lag72', 'dew_temperature_min_lag72',
       'dew_temperature_std_lag72', 'precip_depth_1_hr_mean_lag72',
       'precip_depth_1_hr_max_lag72', 'precip_depth_1_hr_min_lag72',
       'precip_depth_1_hr_std_lag72', 'sea_level_pressure_mean_lag72',
       'sea_level_pressure_max_lag72', 'sea_level_pressure_min_lag72',
       'sea_level_pressure_std_lag72', 'wind_direction_mean_lag72',
       'wind_direction_max_lag72', 'wind_direction_min_lag72',
       'wind_direction_std_lag72', 'wind_speed_mean_lag72',
       'wind_speed_max_lag72', 'wind_speed_min_lag72', 'wind_speed_std_lag72']
# In[21]:


num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)
error = 0

for fold, (train_index, val_index) in enumerate(kf.split(train, target)):

    print ('Training FOLD ',fold,'\n')
    print('Train index:','\tfrom:',train_index.min(),'\tto:',train_index.max())
    print('Valid index:','\tfrom:',val_index.min(),'\tto:',val_index.max(),'\n')
    
    train_X = train[feat_cols].iloc[train_index]
    val_X = train[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(val_X, val_y)
    
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9
            }
    
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=(lgb_train, lgb_eval),
               early_stopping_rounds=20,
               verbose_eval = 20)

    y_pred = gbm.predict(val_X, num_iteration=gbm.best_iteration)
    error += np.sqrt(mean_squared_error(y_pred, (val_y)))/num_folds
    
    print('\nFold',fold,' Score: ',np.sqrt(mean_squared_error(y_pred, val_y)))
    #print('RMSLE: ', rmsle(y_pred, val_y))
    #print('RMSLE_2: ', np.sqrt(mean_squared_log_error(y_pred, (val_y))))

    del train_X, val_X, train_y, val_y, lgb_train, lgb_eval
    gc.collect()

    print (20*'---')
    break
    
print('CV error: ',error)


# In[ ]:


# memory allocation
del train, target
gc.collect()


# ## Prepare Test

# In[ ]:


#preparing test data
building_df = pd.read_csv(PATH+"building_metadata.csv")
test = pd.read_csv(PATH+"/test.csv")
test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
del building_df
gc.collect()


# In[ ]:


#weather_test = pd.read_csv(PATH+"/weather_test.csv")
weather_test = weather_test.drop(drop_cols, axis = 1)
weather_test['timestamp']=weather_test['timestamp'].astype(str)
test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
del weather_test
gc.collect()


# **Reduce Memory**

# In[ ]:


test["primary_use"] = le.transform(test["primary_use"])
test, NAlist = reduce_mem_usage(test);


# **Change dates type**

# In[ ]:


test["timestamp"] = pd.to_datetime(test["timestamp"])
test["hour"] = test["timestamp"].dt.hour.astype(np.uint8)
test["day"] = test["timestamp"].dt.day.astype(np.uint8)
test["weekend"] = test["timestamp"].dt.weekday.astype(np.uint8)
test["month"] = test["timestamp"].dt.month.astype(np.uint8)
test["year"] = test["timestamp"].dt.year.astype(np.uint8)
test = test[feat_cols]


# ## Inference

# In[ ]:


from tqdm import tqdm
i=0
res=[]
step_size = 50000 
for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):
    res.append(np.expm1(gbm.predict(test.iloc[i:i+step_size])))
    i+=step_size


# In[ ]:


del test
gc.collect()


# # Submission

# In[ ]:


res = np.concatenate(res)
sub = pd.read_csv(PATH+"sample_submission.csv")
sub["meter_reading"] = res
sub.to_csv("submission.csv", index = False)
sub.head(10)

