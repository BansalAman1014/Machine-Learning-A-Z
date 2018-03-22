#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

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
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
obj4=StandardScaler()
X_train=obj4.fit_transform(X_train)
X_test=obj4.transform(X_test)

#applying the KNN we get
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)

#predicting the test result
Y_pred=classifier.predict(X_test)

#calculating the confusion matrix
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(Y_test,Y_pred)

#visualsing the training results
from matplotlib.colors import ListedColormap
X_set,Y_set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('KNN(Train Set)')
plt.xlabel('Age')
plt.ylabel('salary')
plt.legend()
plt.show()

#visualsing the test results
from matplotlib.colors import ListedColormap
X_set,Y_set=X_test,Y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('KNN(Test Set)')
plt.xlabel('Age')
plt.ylabel('salary')
plt.legend()
plt.show()