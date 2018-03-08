import numpy as np
import matplotlib as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,train_size=0.8,random_state=0)

"""from sklearn.preprocessing import StandardScaler
obj4=StandardScaler()
X_train=obj4.fit_transform(X_train)
X_test=obj4.transform(X_test)"""