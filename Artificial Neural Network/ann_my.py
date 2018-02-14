# Artificial Neural Network

#importigng libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset= pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Spliting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state= 0 )

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

# Importing Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense

 

# Initiasing the ANN
classifier= Sequential()

# Adding the first input layer and the first hidden layer 
classifier.add(Dense(units=6, activation="relu", input_dim=11, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train,Y_train, batch_size= 10, epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Homework 
New_Prediction= classifier.predict(sc_X.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
New_Prediction= (New_Prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


#Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    classifier= Sequential()
    classifier.add(Dense(units=6, activation="relu", input_dim=11, kernel_initializer="uniform"))
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier, batch_size= 10, epochs=100)
#if __name__ == "__main__":
#accuracies = cross_val_score(estimator=classifier, X = X_train, y = Y_train, cv = 10, n_jobs= 1)
accuracies= cross_val_score(estimator= classifier, X= X_train, y= Y_train, cv= 10, n_jobs= 1)
mean=accuracies.mean()
variance=accuracies.std()
# Improving the ANN

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier= Sequential()
    classifier.add(Dense(units=6, activation="relu", input_dim=11, kernel_initializer="uniform"))
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier)
parameters= {'batch_size': [25, 32], 
             'epochs': [100,250],
             'optimizer': ['adam', 'rmsprop']}
grid_search= GridSearchCV(estimator= classifier, param_grid= parameters, scoring= 'accuracy', cv=10)
grid_search= grid_search.fit(X_train,Y_train)
best_parameters =grid_search.best_params_
best_accuracy = grid_search.best_score_ 


