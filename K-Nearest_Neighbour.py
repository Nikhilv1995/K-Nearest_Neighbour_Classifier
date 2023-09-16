# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:24:34 2023

@author: nikhilve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Reading the CSV data
data = pd.read_csv('Social_Network_Ads.csv')

#selecting features and result.   OR Vector of DV(Dependent Variables) y, and Matrix of IV(Independent Variables) x
x=data.iloc[:,[2,3]].values
y=data.iloc[:,[4]].values
#y=y.drop(['Purchased_No'],axis=1)

#Feature Scaling- Here we are scaling all the data into the same scale.
#can be done before train test split, then it would be easy.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


#Using train-test split to break the data into training and testing data. test_size= 20%data is reserved for testing  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

#Data Pre-Processing part is done till here
#Implementing KNN algo for classification problem

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=17)


#Training the model
y_train = y_train.ravel()
classifier.fit(x_train, y_train)



#Using trained model for performing predictions.
y_pred = classifier.predict(x_test)

#using pickle method
filename = 'model.pkl'
pickle.dump(classifier, open(filename,'wb'))

filename.close()

loaded_model=pickle.load(open('model.pkl','rb'))




#Checking the accuracy of our model using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

print(cm)
#print("Prediction :",classifier.predict([[23,1090997]]))#passing age and salary and predicting using the model.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
