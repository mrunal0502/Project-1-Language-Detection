#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")


# In[2]:


print(data.head())


# In[3]:


data.isnull().sum()


# In[4]:


data["language"].value_counts()


# In[5]:


x=np.array(data["Text"])
y=np.array(data["language"])


# In[ ]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[8]:


model= MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test,y_test)


# In[ ]:


user = input("Enter a text:    ")


# In[ ]:


data = vectorizer.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[ ]:




