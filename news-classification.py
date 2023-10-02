#!/usr/bin/env python
# coding: utf-8

# # CODE :

# In[1]:


pip install nltk


# In[2]:


import nltk


# In[3]:


nltk.download('punkt')


# # IMPORT DATASET

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


fake=pd.read_csv("Fake.csv")

genuine=pd.read_csv("True.csv")


# In[6]:


fake


# In[7]:


display(fake.info())


# In[8]:


display(genuine.info())


# In[9]:


display(genuine.head())


# In[10]:


display(fake.head(23509))


# In[11]:


print(fake.subject.value_counts())


# In[12]:


fake['target']=0
genuine['target']=1


# In[13]:


display(genuine.head())


# In[14]:


display(fake.head(10))


# In[15]:


data=pd.concat([fake,genuine],axis=0)


# In[16]:



#data= pd.DataFrame(data, columns = ['title', 'text', 'subject', 'date'])
data


# In[17]:


data=data.reset_index(drop=True)


# In[18]:


data=data.drop(['title','subject','date'],axis=1)


# In[19]:


print(data.columns)


# # # DATA PREPROCESSING

# # TOKENIZATION

# In[20]:


from nltk.tokenize import word_tokenize


# In[21]:


data['text']=data['text'].apply(word_tokenize)


# In[22]:


print(data.head(10))


# # STEMMING

# In[23]:


from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
porter =SnowballStemmer("english",ignore_stopwords=False)


# In[24]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[25]:


data['text']=data['text'].apply(stem_it)


# In[26]:


#print(data.head(10))
data


# # STOPWORD REMOVAL

# In[27]:


from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words('english'))


# In[28]:


def stop_it(t):
    dt=[word for word in t if len(word)>>2]
    return dt


# In[29]:


data['text']=data['text'].apply(stop_it)


# In[30]:


print(data.head(10))


# In[31]:


data['text']=data['text'].apply(' '.join)


# In[32]:


data


# # SPLITTING UP OF DATA

# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data['text'],data['target'],test_size=0.25)
display(X_train.head())
print('\n')
display(y_train.head())


# # VECTORIZATION

# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(max_df=0.7)

tfidf_train=tfidf.fit_transform(X_train)
tfidf_test=tfidf.transform(X_test)


# In[35]:


print(tfidf_train)


# # BUILDING OF ML MODEL

# # LOGIC REGRESSION
# 

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[37]:


model_1=LogisticRegression(max_iter=900)
model_1.fit(tfidf_train,y_train)


# In[38]:


pred_1= model_1.predict(tfidf_test)


# In[39]:


pred_1


# In[40]:


y_test


# In[41]:


cr1=accuracy_score(y_test,pred_1)
cr1


# In[42]:


from sklearn.linear_model import PassiveAggressiveClassifier

model=PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)


# In[43]:


y_pred=model.predict(tfidf_test)
accscore=accuracy_score(y_test,y_pred)
print(accscore)


# In[ ]:




