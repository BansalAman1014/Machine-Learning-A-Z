# Import the following packages pandas, numpy and matplotlib.
# A Simple Linear Regression is used to predict the salary according to the work experience.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset '.../Salary_Data.csv' by adding path file.
dataset=pd.read_csv('Salary_Data.csv')

#Make the data Between dependent data(Y) and the Independent data(X)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

# Split the training and testing data in required ratio at test_size.
#Import the sklearn package and use the class train_test_split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)

# Perform Linear Regressiona dn fit the training examples in the model and predict the testing data.
# reg is defind as the object for the LinearRegression class.
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

# Will store the prdeiction in the Y_prdeiction
Y_predicition=reg.predict(X_test)

# Can view the salary of the employees which have there experience in the X_test and they are quite close to Y_test values.
print(Y_test)
print(Y_predicition)

#Can view the plotted scatter graph which shows that all the train data is very much near to the predicted value(blue color)
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title('Salary vs experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()