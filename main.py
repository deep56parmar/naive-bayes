#!/usr/bin/env python
# coding: utf-8

# # Gaussian Naive Bayes #
# 
# ## Implementation Using Python 3 ##
# 
# 
# 

# In[24]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report


# ### Loading Datasets ###

# In[25]:


cancer_dataset = load_breast_cancer()
diabetes_dataset = load_diabetes()
iris_dataset = load_iris()
wine_dataset = load_wine()


# ### Data Preprocessing ###

# #### Cancer Dataset ####

# In[26]:


cancer_df = pd.DataFrame(cancer_dataset.data, columns = cancer_dataset.feature_names)
cancer_df['target'] = cancer_dataset.target
print(cancer_df.head())
cancer_x_train, cancer_x_test, cancer_y_train, cancer_y_test = train_test_split(cancer_dataset.data,
                                                                               cancer_dataset.target,
                                                                                test_size = 0.4,
                                                                               random_state = 0)


# #### Diabetes Dataset ####

# In[26]:


diabetes_df = pd.DataFrame(diabetes_dataset.data, columns = diabetes_dataset.feature_names)
diabetes_df['target'] = diabetes_dataset.target
print(diabetes_df.head())
diabetes_x_train, diabetes_x_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_dataset.data,
                                                                               diabetes_dataset.target,
                                                                                test_size = 0.4,
                                                                               random_state = 0)



# #### Wine Dataset ####

# In[26]:


wine_df = pd.DataFrame(wine_dataset.data, columns = wine_dataset.feature_names)
wine_df['target'] = wine_dataset.target
print(wine_df.head())
wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(wine_dataset.data,
                                                                               wine_dataset.target,
                                                                                test_size = 0.4,
                                                                               random_state = 0)

# #### Iris Flower Dataset ####

# In[27]:


iris_df = pd.DataFrame(iris_dataset.data, columns = iris_dataset.feature_names)
iris_df['target'] = iris_dataset.target
print(iris_df.head())
iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_dataset.data,
                                                                               iris_dataset.target,
                                                                                test_size = 0.4,
                                                                               random_state = 0)


# ### Implementation Using SCIKIT's inbuilt function ###

# In[30]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
cancer_y_pred = gnb.fit(cancer_x_train, cancer_y_train).predict(cancer_x_test)
print("Number of mislabeled points out of a total %d points : %d in cancer Dataset" % (cancer_x_test.shape[0], (cancer_y_test != cancer_y_pred).sum()))

diabetes_y_pred = gnb.fit(diabetes_x_train, diabetes_y_train).predict(diabetes_x_test)
print("Number of mislabeled points out of a total %d points : %d in diabetes Dataset" % (diabetes_x_test.shape[0], (diabetes_y_test != diabetes_y_pred).sum()))

wine_y_pred = gnb.fit(wine_x_train, wine_y_train).predict(wine_x_test)
print("Number of mislabeled points out of a total %d points : %d in wine Dataset" % (wine_x_test.shape[0], (wine_y_test != wine_y_pred).sum()))

iris_y_pred = gnb.fit(iris_x_train, iris_y_train).predict(iris_x_test)
print("Number of mislabeled points out of a total %d points : %d in Iris Flower Dataset" % (iris_x_test.shape[0], (iris_y_test != iris_y_pred).sum()))

