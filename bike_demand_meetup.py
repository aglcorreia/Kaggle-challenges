# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:50:40 2018

@author: Antonio
"""

import os
os.chdir('C:\\Users\\Antonio\\Desktop\\KAGGLE MEETUP\\bike sharing demand')

import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')

# convert date to datetime, extract date info and set it to index
train.datetime=pd.to_datetime(train.datetime)

train=train.set_index("datetime")

# some feature engineering (extract hour, day, month, year, weekday)
train["hour"]=train.index.hour
train["day"]=train.index.day
train["month"]=train.index.month
train["year"]=train.index.year
train["weekday"]=train.index.weekday

# check for missing values
train.isnull().sum() # no missing values within the given variables

# missing dates?
train["hour"].groupby(train["year"]).count() # yep
train["hour"].groupby([train.year,train.month]).count() # mostly january

train["hour"][train["year"]==2011][train["month"]==1].groupby([train.day]).count()
train["hour"][train["year"]==2011][train["month"]==2].groupby([train.day]).count()
train["hour"][train["year"]==2011][train["month"]==3].groupby([train.day]).count()

train[train["year"]==2011][train["month"]==1][train["day"]==18]

"""

# add missing dates

train.head()

idx = pd.date_range('01-01-2011 00:00:00', '31-12-2012 23:00:00',freq='H')

train = train.reindex(idx)

train.head()
del idx

# interpolate missing values
train["count"]=train["count"].interpolate(method='time')

train["hour"]=train.index.hour
train["day"]=train.index.day
train["month"]=train.index.month
train["year"]=train.index.year
train["weekday"]=train.index.weekday

"""

# try to predict something
rf = RandomForestClassifier(n_estimators=100)

trainX = train.copy()
del trainX["count"]; del trainX["registered"]; del trainX["casual"]

Y_casual = train["casual"]
Y_reg = train["registered"]

rf.fit(trainX, Y_casual)

# import test
test = pd.read_csv('test.csv')

test.datetime=pd.to_datetime(test.datetime)

test=test.set_index("datetime")

# some feature engineering (extract hour, day, month, year, weekday)
test["hour"]=test.index.hour
test["day"]=test.index.day
test["month"]=test.index.month
test["year"]=test.index.year
test["weekday"]=test.index.weekday

pred_casual = rf.predict(test)

# fit the registered
rf.fit(trainX, Y_reg)

pred_reg = rf.predict(test)

count = pred_casual + pred_reg

datetime=test.index.to_series()

submission = pd.DataFrame({'datetime' : datetime,'count' : count})

submission = submission.set_index(submission.datetime)

del submission["datetime"]

submission.to_csv('submission.csv')


