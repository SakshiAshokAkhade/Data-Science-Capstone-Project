#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[24]:


df=pd.read_csv("CAR DETAILS.csv")
df.head()


# In[25]:


df.shape


# In[26]:


df.duplicated().sum()


# In[27]:


df=df.drop_duplicates()
df.duplicated().sum()
df.shape


# In[28]:


df.isnull().sum()


# In[29]:


df.info()


# In[30]:


df.describe()


# # One-Hot Encoding

# In[31]:


# One-Hot Encoding for categorical variables
df= pd.get_dummies(df, columns=['name','fuel', 'seller_type', 'transmission', 'owner'])


# In[32]:


df.head()


# # Scaling

# In[33]:


# Define the target variable and feature variables
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the numerical columns and transform them
X[['km_driven']] = scaler.fit_transform(X[['km_driven']])
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()


# # Preparing data for Machine Learning Modeling

# In[34]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting arrays
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# In[35]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 


# # Linear Regression

# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = 'CAR DETAILS.csv'
df = pd.read_csv(file_path)

# Data preprocessing
df.drop_duplicates(inplace=True)
df = pd.get_dummies(df, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'])

# Define the target variable and feature variables
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the numerical columns and transform them
X[['km_driven']] = scaler.fit_transform(X[['km_driven']])
y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluation
lr_mae = mean_absolute_error(y_test, y_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lr_r2 = r2_score(y_test, y_pred)

print(f"Linear Regression - MAE: {lr_mae:.4f}, RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")


# # Decision Tree Regressor

# In[37]:


from sklearn.tree import DecisionTreeRegressor

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Evaluation
dt_mae = mean_absolute_error(y_test, y_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
dt_r2 = r2_score(y_test, y_pred)

print(f"Decision Tree Regressor - MAE: {dt_mae:.4f}, RMSE: {dt_rmse:.4f}, R²: {dt_r2:.4f}")


# # Random Forest Regressor

# In[38]:


from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(y_test, y_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rf_r2 = r2_score(y_test, y_pred)

print(f"Random Forest Regressor - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")


# In[ ]:





# #  Gradient Boosting Regressor

# In[39]:


from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

# Evaluation
gbr_mae = mean_absolute_error(y_test, y_pred)
gbr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
gbr_r2 = r2_score(y_test, y_pred)

print(f"Gradient Boosting Regressor - MAE: {gbr_mae:.4f}, RMSE: {gbr_rmse:.4f}, R²: {gbr_r2:.4f}")


# # Support Vector Regressor(SVR)

# In[40]:


from sklearn.svm import SVR

# Support Vector Regressor (SVR)
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

# Evaluation
svr_mae = mean_absolute_error(y_test, y_pred)
svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
svr_r2 = r2_score(y_test, y_pred)

print(f"Support Vector Regressor - MAE: {svr_mae:.4f}, RMSE: {svr_rmse:.4f}, R²: {svr_r2:.4f}")


# # Model Comparison Code

# In[41]:


# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    results[name] = {
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R^2": train_r2,
        "Test R^2": test_r2
    }

# Print the results
for name, metrics in results.items():
    print(f"{name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")


# # Saving Model

# In[42]:


import pickle
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)

# Save the model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest, file)
print("Model saved successfully!")


# In[43]:





# # 20 data points from the CAR DETAILS dataset

# In[44]:


import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
car_data = pd.read_csv('CAR DETAILS.csv')

# One-hot encode categorical columns including 'name'
car_data_encoded = pd.get_dummies(car_data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'])

# Randomly select 20 data points from the original dataset
random_20_data = car_data.sample(n=20, random_state=42)

# One-hot encode the random 20 data points including 'name'
random_20_data_encoded = pd.get_dummies(random_20_data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'])

# Align the columns with the full dataset's columns used for training
required_columns = car_data_encoded.drop('selling_price', axis=1).columns
random_20_data_encoded = random_20_data_encoded.reindex(columns=required_columns, fill_value=0)

# Define the feature variables and target variable for the random 20 data points
X_random_20 = random_20_data_encoded
y_random_20 = random_20_data['selling_price']

# Load the scaler used during training (if available) or initialize a new one
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit and transform the entire training dataset (not shown in your code)
# Assuming you have the training data used during model fitting, use it here
X_train_encoded = car_data_encoded.drop('selling_price', axis=1)
y_train = car_data_encoded['selling_price']

scaler_X.fit(X_train_encoded[['km_driven']])
scaler_y.fit(y_train.values.reshape(-1, 1))

# Transform the random 20 data points
X_random_20[['km_driven']] = scaler_X.transform(X_random_20[['km_driven']])
y_random_20 = scaler_y.transform(y_random_20.values.reshape(-1, 1)).flatten()

# Load the saved model using pickle
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Use the loaded model to make predictions
y_pred_random_20 = loaded_model.predict(X_random_20)

# Inverse transform the predicted and actual values to get them back to original scale
y_pred_random_20_inverse = scaler_y.inverse_transform(y_pred_random_20.reshape(-1, 1)).flatten()
y_random_20_inverse = scaler_y.inverse_transform(y_random_20.reshape(-1, 1)).flatten()

# Evaluate the model
random_20_mse = mean_squared_error(y_random_20_inverse, y_pred_random_20_inverse)
random_20_mae = mean_absolute_error(y_random_20_inverse, y_pred_random_20_inverse)
random_20_r2 = r2_score(y_random_20_inverse, y_pred_random_20_inverse)

print("Random 20 Data Points Results:")
print(f"MSE: {random_20_mse}")
print(f"MAE: {random_20_mae}")
print(f"R^2: {random_20_r2}")


# In[45]:


# Save random 20 data points as an Excel file
random_20_data_encoded.to_excel('random_20_data_encoded.xlsx', index=False)


# In[ ]:




