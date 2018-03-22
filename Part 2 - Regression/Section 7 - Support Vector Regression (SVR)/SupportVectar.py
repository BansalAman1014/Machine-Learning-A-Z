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
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)


#fitting SVR to the dataset
from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(X,Y)

#prdeicting a new result
Y_pred= sc_Y.inverse_transform(reg.predict(sc_X.transform(np.array([[6.5]]))))

#visualizing the dataset
plt.scatter(X,Y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.title('Truth or bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()