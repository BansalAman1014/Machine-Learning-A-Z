#import libraries
import numpy as np
import matplotlib as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:4].values
Y=dataset.iloc[:,4].values

#categories
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
obj1=LabelEncoder()
X[:,3]=obj1.fit_transform(X[:,3])
obj3=OneHotEncoder(categorical_features=[3])
X=obj3.fit_transform(X).toarray()

#Avoiding the dummy variable
X=X[:,1:]
#train and test split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
obj4=StandardScaler()
X_train=obj4.fit_transform(X_train)
X_test=obj4.transform(X_test)"""

#fitting multiple linear regression to the model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

#predicting the values of the model
Y_prediction=reg.predict(X_test)

reg.score(X_test,Y_test)
#Applying for the backward elimination to the model
import statsmodels.formula.api as sm
X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
reg_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
reg_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,3,4,5]]
reg_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,3,5]]
reg_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
reg_OLS.summary()

X_opt=X[:,[0,3]]
reg_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
reg_OLS.summary()

from sklearn.cross_validation import train_test_split
X_train1,X_test1,Y_train1,Y_test1= train_test_split(X_opt,Y,test_size=0.2,random_state=0)

#After applying the backward elimination
from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(X_train1,Y_train1)

#predicting the values of the model
Y_prediction=reg.predict(X_test1)

reg1.score(X_test1,Y_test1)