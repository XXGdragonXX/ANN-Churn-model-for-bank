# -*- coding: utf-8 -*-

# Artificial Neural Network

### Importing the libraries


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

## Part 1 - Data Preprocessing

### Importing the dataset


dataset = pd.read_csv('Churn_Modelling.csv')

dataset.head()

X = dataset.drop(columns = ["Exited","RowNumber","CustomerId","Surname"])

print(X)


y = dataset["Exited"]

print(y)

### Encoding categorical data

#Label Encoding the "Gender" column

X = pd.get_dummies(X, columns=['Gender'], prefix=['Gender'])

X = X.rename(columns={'Gender_Female': 'is_female'})
X = X.rename(columns={'Gender_Male': 'is_male'})

print(X)

"""One Hot Encoding the "Geography" column"""

X = pd.get_dummies(X, columns=['Geography'], prefix=['Geography'])

print(X)

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train

"""### Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train

## Part 2 - Building the ANN

### Initializing the ANN

'''

Variable to create ANN
sequential is a class taken from the tensorflow keras models.
ann is an instance of the sequential of the class

'''

ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer

the sequential class has an add method which we will use to add any
method from the sequential class . we are addind the fully connected
Dense class which creates the fully automatic connected layer and adds
the input layer as well
units : no of hidden neurons in one layer which is added experimentally
actiavtion : activation funtion to be used , in this case we have used
Rectifier activation function aka relu"""

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""### Adding the second hidden layer"""

# this is the second layer which is same as the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""### Adding the output layer"""

# In the output layer we have only one output ( as binary variable)
# hence unit = 1
# activation used is sigmoid as we have to find the probabltiy
# if the customer exiting as well as the prediction
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

# for training the ann we are going to use the method complile
# optimizer : are algorithms or methods used to adjust the parameters of a model
# in order to minimize a loss function or objective function. here we are using
# the adam optimizer which performs atochastic gradient descent
# loss : gives the loss function to be used . here we are using binary-CROSSENTROPY
# for categorical we have to use categorical_entropy
# metrics : used to evaluvate the model . here we are using accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the ANN on the Training set"""

# the method for training we are using fit method
# batch size : the size of the batch we are using as one iteration .
# epochs : no of times the whole dataset runs to improve the accuracy over time
# you see during training the accuracy is increasing over time.
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

"""## Part 4 - Making the predictions and evaluating the model"""

# the input values will be in the order CreditScore	Age	Tenure	Balance
# NumOfProducts	HasCrCard	IsActiveMember EstimatedSalary	is_female	is_male
# Geography_France	Geography_Germany	Geography_Spain
# for prediction we the predcict method from the sequential class and then transform the values
# checking for 100 test cases
print("-------- prdicting the value for the ann----------\n")
list = []
truth = False
# xtest = X_test[:100,:]
xtest = X_test
for val in xtest:
  val = val.reshape(1, 13)
  pred = ann.predict(val)
  if pred > 0.5 :
    truth = True
  else :
    truth = False
  list.append(truth)

ytest = y_test

"""
Making the confusion matrix from the test results

"""

dict = {

    "predicted" : list,
    "actual" : ytest
}
data = pd.DataFrame(dict)
data['predicted'] = data['predicted'].astype(int)
data.head()
tp , fp , tn , fn = 0 , 0 , 0 , 0
data1 = data.values.tolist()
for d in data1 :
  if d[0] == 1 and d[0] == d[1] :
    tp = tp + 1
  elif d[0] == 1 and d[0] != d[1] :
    fp = fp + 1
  elif d[0] == 0 and d[0] != d[1] :
    fn = fn + 1
  elif d[0] == 0 and d[0] == d[1] :
    tn = tn + 1
mat = np.array([[tp,fp],[fn,tn]])
print(mat)

"""Therefore, our ANN model predicts that this customer stays in the bank!

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

Creating the final dataframe of the validation values
"""
og_data = sc.inverse_transform(xtest)
og_data = og_data.tolist()
datax = pd.DataFrame(columns=X.columns , data = og_data)
data = data.reset_index(drop=True)
datax = datax.reset_index(drop=True)
# Merge the dataframes on the common key
result = pd.concat([datax, data], axis = 1)
print(result.head())

