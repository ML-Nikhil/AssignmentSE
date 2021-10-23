# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:57:58 2021

@ author: Nikhil J
@ github : ML-Nikhil

Problem Statement: To perform the logistic regression on Purchase_History data,
as to predict the Purchased variable.

"""

# Importing Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing dataset
df = pd.read_csv('Purchase_History.csv')

# Data Visualization
f = sns.FacetGrid(df,col='Purchased',row='Gender')
f.map(plt.hist,'Age',alpha=0.5,bins=10)
# In Male : After approx 48 yrs everyone has purchased , female has mixed feature
# In Female : Purchase frequency is higher age:45-50 , as compared to male
g = sns.FacetGrid(df,col='Purchased',row='Gender')
g.map(plt.hist,'EstimatedSalary',alpha=0.5,bins =5)

sns.countplot(x='Gender',hue = 'Purchased',data =df) #Fwmale More Purchases
sns.countplot(x='Age',hue="Purchased",data =df) # Purchase is starting from age 26
plt.hist('Purchased',data=df)

# Gender is categorical Variable here. Converting to Dummies

pd.get_dummies(df['Gender'],drop_first=True) # Create seperate Female and Male cols with 0,1 and drops first col
S_dummy = pd.get_dummies(df['Gender'],drop_first=True)


# Adding the dummy col to dataset
df = pd.concat([df,S_dummy],axis=1)
df.head()
# Dropping Gender Column and User ID(not relevant)

df.drop(['Gender','User ID'],axis = 1, inplace = True)

# Applying Logistic Regression
# from Scikitlearn Library

from sklearn.model_selection import train_test_split
X = df.drop('Purchased',axis =1)
Y = df['Purchased']
df.info()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'liblinear')
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
print('accuracy_score of logistic regression model  = ',accuracy_score(y_test, y_pred)*100)
print(logreg.coef_)
print(logreg.intercept_)
df.corr()
# Performing BackElimination Technique
import numpy as np
import statsmodels.api as sm

x_BE = np.append(arr=np.ones((400,1)).astype(int),values=X,axis=1)
y_BE= Y
x_opt = x_BE[:,[0,1,2,3]]
reg_ols = sm.OLS(endog =y_BE, exog = x_opt).fit()
reg_ols.summary()
# since male is insignificant as p-value is greater than 0.05 dropping
x_opt = x_BE[:,[0,1,2]]
reg_ols = sm.OLS(endog = y_BE, exog = x_opt).fit()
reg_ols.summary()

from sklearn.model_selection import train_test_split
x_BE_train,x_BE_test,y_BE_train,y_BE_test = train_test_split(x_opt,y_BE,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
opt_mod = LogisticRegression(solver = 'liblinear')
opt_mod.fit(x_BE_train,y_BE_train)
y_opt_pred = opt_mod.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(y_BE_test,y_opt_pred)

print('Accuracy score is ',accuracy_score(y_BE_test, y_opt_pred)*100)

print(opt_mod.coef_)

#Calculating the intercept:
print(opt_mod.intercept_)

 # equation Purchased ~= 0.108 * Age + 0.0000171 * EstimatedSalary
 # probablity(Purchased) = e^(Purchased)/(1+e^(Purchased))
print('The Gender is not significant for Purchase prediction')












