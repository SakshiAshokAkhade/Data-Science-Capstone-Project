#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[13]:


df=pd.read_csv("CAR DETAILS.csv")
df.head()


# In[14]:


df.duplicated().sum()


# In[15]:


df=df.drop_duplicates()


# In[16]:


df.shape


# In[17]:


# Bar plots for categorical features
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, data=df, ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[18]:


# Example: Count plot of car fuel types
sns.countplot(x='fuel', data=df, palette='viridis')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.title('Count of Cars by Fuel Type')
plt.xticks(rotation=45)
plt.show()


# In[20]:


# Example: Violin plot of car prices by fuel type
sns.violinplot(x='fuel', y='selling_price', data=df, palette='muted')
plt.xlabel('Fuel Type')
plt.ylabel('Price')
plt.title('Violin Plot of Car Prices by Fuel Type')
plt.xticks(rotation=45)
plt.show()


# In[21]:


# Example: Line plot of average car price over years
average_price_by_year = df.groupby('year')['selling_price'].mean().reset_index()
plt.plot(average_price_by_year['year'], average_price_by_year['selling_price'], marker='o')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.title('Average Car Price Over Years')
plt.grid(True)
plt.show()


# In[7]:


# One-Hot Encoding for categorical variables
df= pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'])


# In[ ]:





# In[8]:


# Define the target variable and feature variables
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the numerical columns and transform them
X[['km_driven']] = scaler.fit_transform(X[['km_driven']])
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()


# In[9]:


# Distribution of selling_price
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()


# In[10]:


# Distribution of km_driven
plt.figure(figsize=(10, 6))
sns.histplot(X['km_driven'], bins=30, kde=True)
plt.title('Distribution of Kilometers Driven')
plt.xlabel('Kilometers Driven (scaled)')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# Correlation heatmap
plt.figure(figsize=(18, 8))
corr = X.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[12]:


# 5. Scatter plots of selling_price against numerical features
numerical_columns = ['year', 'km_driven']
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
for i, col in enumerate(numerical_columns):
    sns.scatterplot(x=col, y=y, data=df, ax=axs[i])
    axs[i].set_title(f'Selling Price vs {col}')
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('Selling Price')
plt.tight_layout()
plt.show()


# In[ ]:




