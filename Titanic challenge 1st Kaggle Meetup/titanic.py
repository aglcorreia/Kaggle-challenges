# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:02:32 2018

@author: Antonio
"""

# import libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model as lm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
train.tail()

train.describe()

" check for missing values "
train.info()

" drop irrelevant variables "
train = train.drop(['Ticket','Cabin','Name'], axis=1)
test = test.drop(['Ticket','Cabin','Name'], axis=1)

missing_age = train[train['Age'].isnull()]

missing_age.describe()
train.describe()

" replace missing values in age "
train['Age'].loc[train['Age'].isnull()]=train['Age'].mean()
test['Age'].loc[test['Age'].isnull()]=test['Age'].mean()
train.info()
train.describe()

" do some EDA "

plt.hist(train['Age'].loc[train['Survived']==1],density=True)
plt.hist(train['Age'],density=True)

sns.barplot(x=train['Sex'], y=train['Survived'], data=train);

corr = train.corr() # correlation matrix dataframe
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

" run some liner reg "

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

X_train = train.drop(['Survived','Embarked'], axis=1)
Y_train = train["Survived"]
X_test  = test.drop(['Embarked'], axis=1).copy()
X_test['Fare'].loc[X_test['Fare'].isnull()]=X_test['Fare'].mean()
X_test.info()

logreg = lm.LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

prediction = pd.DataFrame({'PassengerId':test["PassengerId"],'Survived':Y_pred})
prediction.to_csv("titanic_logit.csv", index = False)
