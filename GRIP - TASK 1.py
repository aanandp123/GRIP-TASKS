#!/usr/bin/env python
# coding: utf-8

# The Sparks Foundation Internship - Data Science and Business Analytics Task 1
# Author - Aanand Patel
# Supervised ML Prediction
# Predict the percentage of an student based on the number of study hours.

# 1. Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv("http://bit.ly/w-data")
data.head()


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


plt.scatter(data['Hours'], data['Scores'])
plt.xlabel("Number of Hours")
plt.ylabel("Percentage")
plt.title("Hours vs Percentage")
plt.show()


# Split the data

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['Hours'].values.reshape(-1,1), data['Scores'], test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Training the model

# In[9]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# Predicted Line Plotting
# 

# In[10]:


line = (data['Hours'].values * model.coef_) + model.intercept_
plt.scatter(data.Hours, data.Scores)
plt.plot(data.Hours, line)
plt.show()


# In[15]:


pred = model.predict(X_test)
pred


# Model Evalation
# 

# In[16]:


pred_compare = pd.DataFrame({'Actual Values': y_test, 'Predicted Values':pred})
pred_compare


# In[17]:


from sklearn import metrics
print("Root Mean Squared Error: ", metrics.mean_squared_error(y_test, pred)**0.5)
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, pred))
print("R2 Score: ", metrics.r2_score(y_test, pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, pred))


# CONCLUSION
# 

# In[19]:


h = np.asarray(9.25).reshape(-1,1)
print(model.predict(h)[0])


# In[ ]:




