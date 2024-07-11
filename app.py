import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load your dataset
df = pd.read_csv('CAR_DETAILS.csv')

# Define features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Define the preprocessing steps
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numeric_features = ['km_driven']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline that combines the preprocessor with the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Save the pipeline (which includes the preprocessing and the model)
with open('decision_tree_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
print("Model saved successfully!")
