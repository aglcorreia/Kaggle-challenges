# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:50:47 2018

@author: Antonio
"""

import os
os.chdir('C:\\Users\\Antonio\\Desktop\\KAGGLE MEETUP\\model evaluation\\Digit Recognizer\\data')

import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_no_zero_cols = train[[col for col in train.columns if train[col].sum() != 0]]

train.drop(columns='label')

X_train, X_test, y_train, y_test = train_test_split(train.drop(columns='label'), train.label, stratify=train.label,test_size=0.33)

clf = KNeighborsClassifier(n_neighbors=8)

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

(preds == y_test).sum() / y_test.size
