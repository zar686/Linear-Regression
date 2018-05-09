
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("USA_Housing.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[8]:


df.columns


# In[9]:


sns.pairplot(df)


# In[10]:


sns.distplot(df["Price"])


# In[12]:


sns.heatmap(df.corr(), annot=True)


# In[13]:


df.columns


# In[14]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[15]:


y = df["Price"]


# In[17]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=101)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lm = LinearRegression()


# In[22]:


lm.fit(X_train, y_train)


# In[23]:


print(lm.intercept_)


# In[24]:


lm.coef_


# In[25]:


X_train.columns


# In[26]:


cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coeff"])


# In[27]:


cdf


# In[28]:


predictions = lm.predict(X_test)


# In[29]:


predictions


# In[30]:


plt.scatter(y_test, predictions)


# In[31]:


sns.distplot((y_test - predictions))


# In[32]:


from sklearn import metrics


# In[33]:


metrics.mean_absolute_error(y_test,predictions)


# In[34]:


metrics.mean_squared_error(y_test,predictions)


# In[36]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))

