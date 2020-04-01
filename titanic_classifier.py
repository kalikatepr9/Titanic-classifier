#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:30:39 2020

@author: pranavkalikate
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

#Exploratory Data Analysis

#get the numeric features from the dataset
numeric_features_train = train_df.select_dtypes(include=[np.number]) #include all numeric features
numeric_features_test = test_df.select_dtypes(include=[np.number])

#getting the categorical features and its description
categorical_features = train_df.select_dtypes(exclude=[np.number]) #exclude all numeric features
categorical_feature_description=categorical_features.describe()

#Check for missing values in train.csv and test.csv files
null_train = pd.DataFrame(numeric_features_train.isnull().sum().sort_values(ascending=False)[:80]) 
null_test = pd.DataFrame(numeric_features_test.isnull().sum().sort_values(ascending=False)[:80])

#Correlation matrix
corr_matrix=train_df.corr()  #.abs()
corr_sales=corr_matrix['SalePrice'].sort_values(ascending=False)[:38] #correlated features with SalesPrice

#Getting the features with null values
nulls_1 = pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False)[:25]) #features with null values
nulls_1.columns = ['Null Count']
nulls_1.index.name = 'Feature'

nulls_2 = pd.DataFrame(test_df.isnull().sum().sort_values(ascending=False)[:25]) #features with null values
nulls_2.columns = ['Null Count']
nulls_2.index.name = 'Feature'

#Taking care of categorical data
"""from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[4])], remainder='passthrough') #passthrough to concatenate
train=np.array(ct.fit_transform(train), dtype=np.float)
"""
#Encode Sex
train_df['enc_Sex'] = pd.get_dummies(train_df.Sex, drop_first=True)
test_df['enc_Sex'] = pd.get_dummies(test_df.Sex, drop_first=True)
print ('Encoded: \n')
print (train_df.enc_Sex.value_counts())

#Encode Embarked
print ("Original: \n")
print (train_df.Embarked.value_counts(), "\n")
def encode(x):
    return 1 if x == 'S' else 0  #since S has come highest no. of times = encoded 1
train_df['enc_Embarked'] = train_df.Embarked.apply(encode) #assign 1 for S otherwise 0
test_df['enc_Embarked'] = test_df.Embarked.apply(encode)
print ('Encoded: \n')
print (train_df.enc_Embarked.value_counts())

#Getting final numeric features for modelling
train = train_df.select_dtypes(include=[np.number]).drop(['PassengerId'], axis=1)
test= test_df.select_dtypes(include=[np.number]).drop(['PassengerId'], axis=1)

#Missing values
train_missing_values=train.isnull().sum()
test_missing_values=test.isnull().sum()

#Taking care of missing data in train & test set
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
train.iloc[:,[2]]=imputer.fit_transform(train.iloc[:,[2]])  #iloc since dataframe
test.iloc[:,[1,4]]=imputer.fit_transform(test.iloc[:,[1,4]])

#check if missing values are replaced or not
train.isnull().sum()
test.isnull().sum()

#Building a Model
X=train.iloc[:,1:8].values  #Dependent feature
y=train.iloc[:,0].values   #Independent feature

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

"""X_test = X_test.as_matrix()
X_train = X_train.as_matrix()"""

#Fitting the model to training set
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
model=classifier.fit(X_train, y_train)

"""# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# Applying k-Fold Cross Validation (model evaluation)  
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std() 

# Predicting the Test set results
#probabilities as output
y_proba = classifier.predict_proba(test.values) 
predictions=classifier.predict(test.values)

#Getting a csv file
output=pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':predictions})
output.to_csv('titanic_XGB.csv', index=False)