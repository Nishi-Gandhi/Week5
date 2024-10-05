#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd


# In[2]:


# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')
titanic.head()


# In[3]:


# Drop irrelevant columns for simplicity
titanic = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']].dropna()

# Convert categorical columns to numeric
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Define the features and target
X = titanic.drop('survived', axis=1)
y = titanic['survived']


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)


# In[23]:


# Save the trained model
path = "titanic_model.pkl"
with open(path, 'wb') as model_file:
    pickle.dump(clf, model_file)
    
from google.cloud import storage

blob = storage.blob.Blob.from_string('gs://gcloud-pipelines/titanic_model.pkl', client = storage.Client())
blob.upload_from_filename('titanic_model.pkl')


# In[ ]:




