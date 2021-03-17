# Basically we are predicting whether an customer will exit from bank or not based on certain
# parameters
# Step -1 -> Data processing

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset

dataset = pd.read_csv('Churn_Modelling.csv')
#/content/drive/MyDrive/ANN/Churn_Modelling.csv - For colab

X = dataset.iloc[:,3:13] 
y = dataset.iloc[:,13]

# create dummy variables

geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

# concatinate the Data frames

X = pd.concat([X,geography,gender],axis = 1)

# Drop unnecessary columns

X = X.drop(['Geography','Gender'],axis = 1)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step - 2-> Now lets make ANN

# Importing the Keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# initializing the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units= 6, kernel_initializer = 'he_uniform',activation = 'relu',input_dim = 11))

# Adding the second hidden layer

classifier.add(Dense(units= 6, kernel_initializer = 'he_uniform',activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units= 1, kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history = classifier.fit(X_train,y_train,validation_split = 0.33, batch_size = 10, epochs = 100)


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)