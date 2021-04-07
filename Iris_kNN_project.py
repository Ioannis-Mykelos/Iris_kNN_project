#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


# Load the Iris data set
from sklearn.datasets import load_iris
iris_dataset=load_iris()
print(iris_dataset)


# In[36]:


print('Keys of the iris_dataset:\n',iris_dataset.keys())
print('Target names:\n', iris_dataset['target_names'])
print('Feature names:\n', iris_dataset['feature_names'])
print('Shape of the data set:\n', iris_dataset['data'].shape)
print('Target:\n', iris_dataset['target'])


# In[37]:


# Split the data set into train and test data sets.

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

#random_state: makes sure that we will get the same output if we run the same function several times.

print('X_train shape:\n', X_train.shape)
print('y_train shape:\n', y_train.shape)

print('X_test shape:\n', X_test.shape)
print('y_test shape:\n', y_test.shape)


# In[38]:


# Visualization of the data set.
# We will create a pandas dataframe from data in X_train and label the columns using the strings in iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, 
                           alpha=.8, cmap='Reds')


# In[45]:


sns.pairplot(iris_dataframe, vars=iris_dataset['feature_names'], kind='reg', diag_kind='hist')


# In[46]:


# Building k-NN model.
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

# Evaluating the Model
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[47]:


# Sumarry
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




