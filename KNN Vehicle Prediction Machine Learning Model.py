#!/usr/bin/env python
# coding: utf-8

# # Task 1: Developing A Machine Learning Model

# In[273]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# In[274]:


#opening the excel file into a variable called 'data'
data = pd.read_excel(r'/Users/timmys/Desktop/ENDGCure3Dataframe.xlsx')

#creating a dataframe
mdf = DataFrame()
mdf['TypeOfVehicle'] = data.iloc[:, 0]
mdf['Weight(kg)'] = data.iloc[:, 1]
mdf['Horsepower(bhp)'] = data.iloc[:, 2]
mdf['AcccelerationTime(s)'] = data.iloc[:, 3]
mdf


# In[275]:


# splitting the data into a training and test set using SKLEARN
df2 = mdf.drop(columns = ['TypeOfVehicle'])
dy = mdf.TypeOfVehicle
dy


# In[276]:


# splitting the data into a training and test set using SKLEARN
from sklearn.model_selection import train_test_split
df2_train, df2_test, dy_train, dy_test = train_test_split(df2, dy)

df2_train, df2_test, dy_train, dy_test;


# In[277]:


#Data Preprocessing
from sklearn import preprocessing
lenc = preprocessing.LabelEncoder() #defining the label encoder
lenc


# In[278]:


lenc = lenc.fit(dy_train) #fitting the data into the label encoder
lenc.classes_


# In[279]:


#transforming labels into numbers
#our data for the classes (cars and trucks) was already encoded into 0's and 1's (0 for cars, 1 for trucks)
y_train = lenc.transform(dy_train)
dy_train, y_train


# In[280]:


#Preprocessing attributes with MinMaxScaler()
df2_train.to_numpy()


# ## Preprocessing using MinMaxScaler (for Task 2)

# In[281]:


#Defining the preprocessing tool as a variable called 'var2'
var2 = preprocessing.MinMaxScaler()
var2 = var2.fit(df2_train.to_numpy()) #this normalizes the values of the data to range from 0 to 1

x_train = var2.transform(df2_train.to_numpy()) #transforming the attribute dataframe into an array with 
                                               #values from 0 to 1
x_train


# In[282]:


#building the machine learning model - KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5) #defining the classifier, using a k value of 1

knn_model = knn_model.fit(x_train, y_train)
knn_model


# In[283]:


y_test = lenc.transform(dy_test.to_numpy()) #converting dy_test into a numpy array, storing it in variable 'y_test'
x_test = var2.transform(df2_test.to_numpy())#converting df2_test into a numpy array, storing it in variable 'x_test'


# In[284]:


#Applying the model for predictions of x_test and comparing them with y_test

knn_model.predict(x_test)


# In[285]:


y_test


# In[286]:


#Using the 'score' method

knn_model.score(x_test, y_test) #KNN model score for test sets when K = 1


# In[287]:


knn_model.score(x_train, y_train) #KNN model score for training sets when K = 1


# # Task 2: Doing Analysis to Find Best Value of K

# In[288]:


k_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,] #testing k values ranging from 1 to 10
array1 = []
print("Format is [test score, training score]\n")
for value in k_value:
    knn_model = KNeighborsClassifier(value)
    knn_model = knn_model.fit(x_train, y_train)
    score_test = knn_model.score(x_test, y_test) #storing score for test in 'score_test'
    score_train = knn_model.score(x_train, y_train) #storing score for train in 'score_train'
    array = [score_test, score_train]
    array1 += [score_test, score_train] #creating a new array variable that stores all values in a single array
    print(array, "for K value:", value)


# In[289]:


import matplotlib.pyplot as plt #using matplotlib to create the graph that compares score for associated K values
arr1 = DataFrame()
arr1[0] = array1[::2]
arr1[1] = array1[1::2]
arr1[2] = k_value
arr1 


# In[290]:


plt.plot(arr1.iloc[:,2], arr1.iloc[:,0], label = 'Test Scores') #getting score values for test scores
plt.plot(arr1.iloc[:,2],arr1.iloc[:,1], label = 'Training Scores') #getting score values for training scores
plt.xlabel('K Values')
plt.ylabel('Score Values')
plt.title('Test Score VS. Training Score for Various K Values')
plt.legend()
plt.grid(True)


# ## Preprocessing using StandardScaler (for Task 2)

# #### After preprocessing using Standard Scaler the previous for loop and code segment was used to graph the plots as shown in the pdf file submission.

# In[291]:


#Defining the preprocessing tool as a variable called 'var2'
var2 = preprocessing.StandardScaler() #using standardscaler
var2_fitted = var2.fit(df2_train.to_numpy()) #this normalizes the values of the data to range from 0 to 1

x_train_new = var2_fitted.transform(df2_train.to_numpy()) #transforming the attribute dataframe into an array with 
                                                      #values from 0 to 1
print(x_train, "\n")
print(x_train_new, "\n")

x_train_new = (x_train - x_train.mean()) / (x_train.std(ddof=1))

print(x_train_new)


# # Task 3: Finding the Best Value of K and Creating a Confusion Matrix

# In[292]:


#Code for generating the confusion matrix
from sklearn.metrics import confusion_matrix
y_true = y_train[0:25]
y_pred = y_test
confusion_matrix(y_true,y_pred)


# In[293]:


from sklearn.utils.multiclass import unique_labels
unique_labels(y_test)


# In[294]:


import seaborn as sns
def plot (y_true, y_pred):
    labels = unique_labels(y_test)
    column = [f'Predicted{label}' for label in labels]
    indices = [f'Actual{label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true,y_pred), 
                         columns = column, index = indices)

    #return table
    return sns.heatmap(table, annot = True, fmt = 'd', cmap='viridis')


# In[295]:


plot(y_test, y_pred)


# # Task 4: Applying the Model

# In[296]:


#Task 4: Testing a new instance (Applying the model)

print("Final Model Parameters:\n")
print("K value used: 5\n")
print("Scalar used for preprocessing data: MinMaxScaler\n")
print("Model Accuracy in the training set: 0.946\n")
print("Model Accuracy in the test set: 1.0\n\n")

#We are testing the following car and truck (respectively): Chevrolet Corvette Z06 (2020), Mazda BT50 (2020)

#Corvette Z06 (2020) Values for [weight, horsepower, acceleration]: [1655(kg), 670(bhp), 2.6(seconds)]
#Mazda BT50 (2020) Values for [weight, horsepower, acceleration]: [1925(kg), 201(bhp), 9.55(seconds)]

test_value1 = np.array([[1655, 670, 2.6]]) #testing the car
test_value2 = np.array([[1925, 201, 9.55]]) #testing the truck

new1 = var2.transform(test_value1)
new2 = var2.transform(test_value2)

knn_model.predict(new1) #predicting the car 
knn_model.predict(new2) #predicting the truck

print("Expected value: 0,\t predicted value:", knn_model.predict(new1), "\n") 
print("Expected value: 1,\t predicted value:", knn_model.predict(new2), "\n") 


# In[ ]:




