#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#missing values
"""from sklearn.preprocessing import Imputer
obj=Imputer(missing_values='NaN',strategy='mean',axis=0)
obj=obj.fit(X[:,1:3])
X[:,1:3]=obj.transform(X[:,1:3])"""

#categories
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
obj1=LabelEncoder()
obj2=LabelEncoder()
X[:,0]=obj1.fit_transform(X[:,0])
obj3=OneHotEncoder(categorical_features=[0])
X=obj3.fit_transform(X).toarray()
Y=obj2.fit_transform(Y)"""

#train and test split
"""from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,train_size=0.8,random_state=0)"""

#feature scaling
"""from sklearn.preprocessing import StandardScaler
obj4=StandardScaler()
X_train=obj4.fit_transform(X_train)
X_test=obj4.transform(X_test)"""

#Fitting the linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#fitting the polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg1=LinearRegression()
lin_reg1.fit(X_poly,Y)


#visualizing the linear regression
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("truth or bluff(Linear regression)")
plt.xlabel("Position label");
plt.ylabel("Salary")
plt.show()

#visualizing the polynomial regression
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg1.predict(X_poly),color='blue')
plt.title("truth or bluff(Polynomail regression)")
plt.xlabel("Position label");
plt.ylabel("Salary")
plt.show()

#predecting the linear regression result
lin_reg.predict(6.5)

#predecting the polynomial regression result
lin_reg1.predict(poly_reg.fit_transform(6.5))