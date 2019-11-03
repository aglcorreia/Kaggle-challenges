#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import os, warnings, math

from functions_tony import reduce_mem_usage


# In[2]:


train = pd.read_csv('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/train.csv')
test = pd.read_csv('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/test.csv')
meta = pd.read_csv('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/building_metadata.csv')
weather_train = pd.read_csv('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/weather_train.csv')
weather_test = pd.read_csv('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/weather_test.csv')


# In[49]:


train_df = train.set_index('building_id').join(meta.set_index('building_id'), on='building_id', how='left').reset_index()


# In[50]:


train_df = train_df.set_index(['site_id','timestamp']).join(weather_train.set_index(['site_id','timestamp']),      on=['site_id','timestamp'], how='left').reset_index()


# In[51]:


test_df = test.set_index('building_id').join(meta.set_index('building_id'), on='building_id', how='left').reset_index()

test_df = test_df.set_index(['site_id','timestamp']).join(weather_test.set_index(['site_id','timestamp']),      on=['site_id','timestamp'], how='left').reset_index()


# In[38]:


train_df.dtypes


# In[39]:


train_df.describe().T


# In[40]:


reduce_mem_usage(train_df).describe().T


# Downcasting fields affects some of the dataframe's descriptive statistics. We will only downcast the following fields:
# `[site_id','building_id','meter','meter_reading','square_feet']`

# In[41]:


train_df = train.set_index('building_id').join(meta.set_index('building_id'), on='building_id', how='left').reset_index()

train_df = train_df.set_index(['site_id','timestamp']).join(weather_train.set_index(['site_id','timestamp']),      on=['site_id','timestamp'], how='left').reset_index()


# In[72]:


downcast_fields = ['site_id','building_id','meter','meter_reading','square_feet']
train_df_dc_fields = reduce_mem_usage(train_df[downcast_fields])

downcast_fields.remove('meter_reading')
test_df_dc_fields = reduce_mem_usage(test_df[downcast_fields])


# In[88]:


for c in train_df_dc_fields.columns:
    train_df[c] = train_df_dc_fields[c]
    
for c in test_df_dc_fields.columns:
    test_df[c] = test_df_dc_fields[c]


# In[90]:


train_df.to_pickle("/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/train_df.pkl")
test_df.to_pickle("/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/test_df.pkl")


# In[32]:


weather_train[(weather_train['site_id']==15) & (weather_train['timestamp']>='2016-05-05 00:00:00')][0:24]


# Sites which are colder during the day than at night:
#  - [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15]
#  - However, some of these sites are not entirely colder at day than at night. It can be that the real timezone is not the one disclosed on the timestamp field

# In[ ]:




