# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:17:04 2018

@author: pak16
"""
#importing Libraries
#---------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#setting up working directory
#---------------------------------------------------------------------------------------------------------
import os
os.chdir('C:\\Users\\pak16\\Desktop')


# Importing the dataset
#---------------------------------------------------------------------------------------------------------
dataset=pd.read_csv('train.csv')


#basic summary & EDA
#---------------------------------------------------------------------------------------------------------
dataset.describe()
##########Item_Visibility is 0, which is not possible if a product is in supermarket

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#########Item_Weight and Outlet_Size have missing values 


dataset.info()
######## 6 categorical and 4 numerical variables


#Visualizations
#----------------------------------------------------------------------------------------------------------



#Visualizing Numerical data using histograms

plt.hist(dataset['Item_Weight'],range=(0,35),bins=10)


plt.hist(dataset['Item_Visibility'],range=(0,0.35),bins=10)

plt.hist(dataset['Item_MRP'],range=(30,270),bins=10)
######no outliers

sns.boxplot(x="Outlet_Type", y="Item_Outlet_Sales", data=dataset,palette='rainbow',hue='Outlet_Size')
######supermarket3 has huge sales for medium size markets
######there are no small and big markets for most of the outlet types


sns.heatmap(dataset.corr())
####there is slight positive correlation between MRP and Sales


sns.lmplot(x='Item_MRP',y='Item_Outlet_Sales',data=dataset,hue='Outlet_Type',palette='coolwarm')
#####combining knowledge from previos two graphs to create simple regression between MRP and Sales




#Feature Engineering
#----------------------------------------------------------------------------------------------------------
#Outlet_Establishment_Year is not so usefull but outlet age might be useful
#Determining the years of operation of a store using Outlet_Establishment_Year

dataset['Outlet_Establishment_Year']=2016-dataset['Outlet_Establishment_Year']
dataset = dataset.rename(columns={'Outlet_Establishment_Year': 'Outlet_Age'})

g=sns.barplot(x='Item_Type',y='Item_Outlet_Sales',data=dataset,estimator=np.mean)
plt.xticks(rotation=45)
#sales are similiar for all item_types

dataset.drop('Outlet_Identifier',inplace=True,axis=1)
###########Removing Outlet_Identifier which is not useful for analysis

dataset['Item_Type'].value_counts()
######Too many Item_Type, and not much significant difference in sales for each type
#Item type can be combined using Item_Identifier which classifies items into 3 types 
#Encoding using Item_Identifier

#Creating a broad category of Type of Item using Item_Identifier
dataset['Item_Identifier']=dataset['Item_Identifier'].str[:2]

sns.barplot(x='Item_Identifier',y='Item_Outlet_Sales',data=dataset,estimator=np.mean)

dataset['Item_Identifier'].value_counts()

#removing Item_Type from analysis
dataset.drop('Item_Type',inplace=True,axis=1)

dataset['Item_Fat_Content'].value_counts()
########‘Low Fat’ values as mis-spelled as ‘low fat’ and ‘LF’
########‘Regular’ are mis-spelled as ‘regular’

#Modifying categories of Item_Fat_Content
dataset['Item_Fat_Content']=dataset['Item_Fat_Content'].replace({'low fat':'Low Fat',
                                                                    'reg':'Regular',
                                                                    'LF':'Low Fat'})


dataset['Item_Fat_Content'].value_counts()

dataset['Outlet_Type'].value_counts()
dataset['Outlet_Location_Type'].value_counts()
######These two categories don't need any modification

#handling missing values for outlet_size
from sklearn.base import TransformerMixin


class SeriesImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.  

        """
    def fit(self, X, y=None):
        if   X.dtype == np.dtype('O'): self.fill = X.value_counts().index[0]
        else                            : self.fill = X.mean()
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)





a  = SeriesImputer()   # Initialize the imputer
a.fit(dataset['Outlet_Size'])              # Fit the imputer
dataset['Outlet_Size'] = a.transform(dataset['Outlet_Size'])



#checking if imputed properly
dataset['Outlet_Size'].isnull().sum()

#Creating Dummy Variables for Item_Identifier, Item_Fat_Content, Outlet_Type, and Outlet_Location_Type, Outlet_Size
Item_Identifier= pd.get_dummies(dataset['Item_Identifier'], drop_first=True)
Item_Fat_Content= pd.get_dummies(dataset['Item_Fat_Content'], drop_first=True)
Outlet_Type= pd.get_dummies(dataset['Outlet_Type'], drop_first=True)
Outlet_Location_Type= pd.get_dummies(dataset['Outlet_Location_Type'], drop_first=True)
Outlet_Size= pd.get_dummies(dataset['Outlet_Size'], drop_first=True)

dataset= pd.concat([Item_Identifier,Item_Fat_Content,Outlet_Type,Outlet_Location_Type,Outlet_Size,dataset],axis=1)
dataset.drop(['Item_Identifier','Item_Fat_Content','Outlet_Type','Outlet_Location_Type','Outlet_Size'],axis=1,inplace=True)

#Dividing dataset into independent array and dependent vector 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#handling missing values for item_weight
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 10:11])
X[:, 10:11] = imputer.transform(X[:, 10:11])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#OLS
#----------------------------------------------------------------------------------------------------------
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_ols = LinearRegression()
regressor_ols.fit(X_train, y_train)

# Predicting the Test set results
y_pred_ols = regressor_ols.predict(X_test)

#using RMSE for model Evaluation
from sklearn import metrics
OLS_RMSE= np.sqrt(metrics.mean_squared_error(y_test, y_pred_ols))

#DecisionTreeRegression
#----------------------------------------------------------------------------------------------------------
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor()
regressor_dt.fit(X_train, y_train)

# Predicting a new result
y_pred_dt = regressor_dt.predict(X_test)


DecisionTreeRegressor_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred_dt))


#RandomForestRegression
#----------------------------------------------------------------------------------------------------------
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor_rf.fit(X_train, y_train)

# Predicting a new result
y_pred_rf = regressor_rf.predict(X_test)

RandomForestRegressor_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf))


#XGBoost
#----------------------------------------------------------------------------------------------------------
from xgboost import XGBRegressor


xgb =XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, 
                  silent=True, objective='reg:linear', nthread=-1, 
                  gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 
                  colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                  scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

xgb.fit(X_train, y_train)
y_pred_xgboost=xgb.predict(X_test)

XGBRegressor_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgboost))



#-----------------------------------------------------------------------------------------------------------
# Feature Scaling for SVR and ANN
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train=sc_y.fit_transform(y_train.reshape(-1, 1))


#SVR(Gaussian Kernel)
#----------------------------------------------------------------------------------------------------------


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svm = SVR(kernel = 'rbf',C=100,gamma=0.001)
regressor_svm.fit(X_train, y_train)

# Predicting a new result
y_pred_svm = regressor_svm.predict(X_test)
y_pred_svm = sc_y.inverse_transform(y_pred_svm)

SVR_RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred_svm))


#ANN
#----------------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense( 7, input_dim=14 ,activation='relu'))
model.add(Dense( 7, input_dim=14 ,activation='relu'))
model.add(Dense( 3, input_dim=14 ,activation='relu'))
model.add(Dense(1,kernel_initializer ='uniform'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])


model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=False, shuffle=False)
predictions = model.predict(X_test)

predictions = sc_y.inverse_transform(predictions)
ANNRegressor_RMSE=np.sqrt(metrics.mean_squared_error(y_test, predictions))


print("OLS_RMSE is ",  OLS_RMSE)
print("DecisionTreeRegressor_RMSE is ",  DecisionTreeRegressor_RMSE)
print("RandomForestRegressor_RMSE is ",  RandomForestRegressor_RMSE)
print("XGBRegressor_RMSE is ",  XGBRegressor_RMSE)
print("SVR_RMSE is ",  SVR_RMSE)
print("ANNRegressor_RMSE is ",  ANNRegressor_RMSE)

#######Selection Criterion: RMSE

#######Neural Network egressor is the the best Regressor for this dataset with #RMSE=1083.53
