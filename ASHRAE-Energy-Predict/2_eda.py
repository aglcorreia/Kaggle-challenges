#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os, warnings, math
import time
import datetime
from functions_tony import count_uniq_miss
import matplotlib.pyplot as plt
import calendar
from pandas import Grouper
import random


# In[5]:


train_df = pd.read_pickle('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/train_df.pkl')
test_df = pd.read_pickle('/home/antonio.correia/Documents/Kaggle/ASHRAE_energy/test_df.pkl')


# In[6]:


df = pd.concat([train_df,test_df],axis=0)


# In[7]:


df['unix_timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
df['unix_timestamp'] = df['unix_timestamp'].astype(np.int32)


# In[8]:


train = (df['unix_timestamp'] < 1483228800)
test = (df['unix_timestamp'] >= 1483228800)


# # Categorize fields

# In[158]:


df.sample(5).T


# In[9]:


id_fields = ['building_id','site_id']
categoricals = ['primary_use','year_built','floor_count','meter']
numericals = ['air_temperature',
              'cloud_coverage','meter_reading',
              'precip_depth_1_hr',
              'sea_level_pressure','square_feet','wind_direction','wind_speed']


# # Check number of unique values and missing values

# In[9]:


count_uniq_miss(df[id_fields])
count_uniq_miss(df[categoricals])


# # Check category counts for categoricals

# In[18]:


for f in categoricals:
    print("\033[1m" + f + "\033[0m")
    a = df[f].value_counts(dropna=False)
    b = df[f].value_counts(dropna=False,normalize=True)
    pd.concat([a,b],axis=1)
    print(' ')


# ## for `year_built`, visualize categories

# In[12]:


fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax = plt.hist(df['year_built'], bins=40, rwidth=0.9,color='#607c8e')


# ## for `meter`, check which buildings have what
#  - 0: electricity
#  - 1: chilledwater
#  - 2: steam
#  - 3: hotwater

# In[35]:


df[['meter','building_id']].drop_duplicates().groupby(['meter']).count()


#  - Check one example to see how different values between meters are
#  - But first define a function to check which meters a building ID has

# In[91]:


def check_meter(building_id):
    
    if type(building_id)==list or type(building_id)==range:
        n = pd.DataFrame([])
        for i in building_id:
            m = df[['building_id','meter']][df['building_id']==i].drop_duplicates()
            n = pd.concat([m,n],axis=0)
    else:
        n = df[['building_id','meter']][df['building_id']==building_id].drop_duplicates()
    return n.sort_values(by='building_id')


#  - Even more easily, we can pivot the meter category

# In[10]:


n = df[['building_id','meter']].drop_duplicates()
n = n.pivot(index='building_id',columns='meter',values='meter')

for i in range(0,4):
    n = n.replace(to_replace=i,value=1)
    
n = n.fillna(0)

for i in range(0,4):
    n[i] = n[i].astype(int)


# In[144]:


for i in [0,1]:
    for j in [0,1]:
        for k in [0,1]:
            for l in [0,1]:
                print('[0, 1, 2, 3]: ' + str([i,j,k,l]) +' '+ str(n[(n[0]==i) & (n[1]==j) & (n[2]==k) & (n[3]==l)][0].count())+' buildings.')


#  - 852 buildings have meter 0
#  - 220 buildings have meters 0, 1, 2
#  - 132 buildings have meters 0, 1
#  - 111 buildings have meters 0, 2, 3
#  - 64 buildings have meters 0, 2
#  - no building has only meter 3
#  - only a few buildings (36) do not have meter 0
#  - clearly meter 0 (electricity) is the most important source of readings

# ## How different are meter readings between different meters?

# In[169]:


# which buildings have all meters?
all_meters = n[(n[0]==1) & (n[1]==1) & (n[2]==1) & (n[3]==1)].index.tolist()

# check their readings on a random timestamp to see how different are magnitudes
condition = (df['building_id'].isin(all_meters)) & (df['unix_timestamp']==1478376000)

df[['building_id','meter','meter_reading']][condition].pivot(index='building_id',columns='meter',values='meter_reading')


#  - meter readings are quite different from each other in terms of magnitude for different meters

# In[174]:


# check for a given timestamp different meters, this time in a broader group of buildings
condition0 = (df['unix_timestamp']==1478376000) & (df['building_id'].isin(n[(n[0]==1)].index.tolist()))
condition1 = (df['unix_timestamp']==1478376000) & (df['building_id'].isin(n[(n[1]==1)].index.tolist()))
condition2 = (df['unix_timestamp']==1478376000) & (df['building_id'].isin(n[(n[2]==1)].index.tolist()))
condition3 = (df['unix_timestamp']==1478376000) & (df['building_id'].isin(n[(n[3]==1)].index.tolist()))

df[['building_id','meter','meter_reading']][condition0].pivot(index='building_id',columns='meter',values='meter_reading').sample(5)

df[['building_id','meter','meter_reading']][condition1].pivot(index='building_id',columns='meter',values='meter_reading').sample(5)

df[['building_id','meter','meter_reading']][condition2].pivot(index='building_id',columns='meter',values='meter_reading').sample(5)

df[['building_id','meter','meter_reading']][condition3].pivot(index='building_id',columns='meter',values='meter_reading').sample(5)


#  - given the nature of readings, it could make more sense to estimate changes in readings since previous timestamp
#  - should it be percentage changes or absolute changes?
#  - first check how a given meter reading evolves over time
#  - pick a random meter and building id to check this

# In[87]:


def check_random_readings(meter, start_time, num_hours):
    building = random.choice(n[(n[meter]==1)].index.tolist())
    condition = (df['meter']==meter) & (df['building_id']==building) & (df['timestamp']>=start_time)
    return df[condition][['timestamp','site_id','air_temperature','building_id','primary_use','meter','meter_reading']][0:num_hours]

def check_building_readings(building, meter, start_time, num_hours):
    condition = (df['meter']==meter) & (df['building_id']==building) & (df['timestamp']>=start_time)
    return df[condition][['timestamp','site_id','air_temperature','building_id','primary_use','meter','meter_reading']][0:num_hours]


# In[97]:


check_random_readings(1,'2016-05-20 00:00:00',24)


# In[94]:


check_building_readings(1083,0,'2016-06-04 00:00:00',24)


#  - building 865 always seems to have the same reading throughout the day
#  - we should check whether there are other buildings in the same situation

# # Check distributions for numerical variables

# ## `meter_reading`

# In[7]:


df['timestamp'] = pd.to_datetime(df['timestamp'])
ts = df.set_index('timestamp')


# In[8]:


# to plot a random building or site id
def ts_plot_random(num,id_f,site=False):
    
    try:
        n_id = np.random.randint(int(ts[id_f].max()))
    except:
        cats = list(ts[id_f].unique())
        n_id = random.choice(cats)
        
    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        
    if site:
        fig.suptitle(print(str(num)+' for '+str(id_f)+' = '+str(n_id)), fontsize=16)
        axes = ts[ts[id_f]==n_id].groupby(['timestamp'])[num].median().plot()
    else:
        use = ts[ts[id_f]==n_id]['primary_use'][0]
        fig.suptitle(print(str(num)+' for '+str(id_f)+' = '+str(n_id)+', primary use: '+str(use)), fontsize=16)
        axes = ts[ts[id_f]==n_id][num].plot()


# In[88]:


ts_plot_random('meter_reading','building_id')


# In[89]:


ts_plot_random('meter_reading','site_id',site=True)


# In[91]:


ts_plot_random('air_temperature','site_id',site=True)


# In[90]:


ts_plot_random('meter_reading','primary_use',site=True)


# # Check correlations (TBD)

# ## Meter readings across buildings in the same site_id

# In[ ]:




