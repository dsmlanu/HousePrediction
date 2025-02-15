from xml.etree.ElementInclude import include

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

## read CSV file
data=pd.read_csv('Housing.csv')
# Handling missing value
## Drop column where more than 50% values are missing because that column is not useful for training
data = data.dropna(thresh=len(data) * 0.5, axis=1)
## Identifying numerical and categorical features
num_features=data.select_dtypes(include=['int64','float64']).columns.tolist()
cat_features= data.select_dtypes(include=['object']).columns.tolist()
## remove the target feature from the num_features
num_features.remove('SalePrice')
##  handling missing value in numerical column
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # Fill missing values
    ("scaler", StandardScaler())  # Standardize numerical features
])
##Handling missing value in categorical coloumn
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode categorical features
])
# Apply Transformations to Data
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

## split data into train and test data
X = data.drop(columns=["SalePrice"])  # Features
y = data["SalePrice"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# fit preprocesor and Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
# save  preprocessed Data & pipeline
joblib.dump(preprocessor, "preprocessor.pkl")
np.save("X_train.npy", X_train_processed)
np.save("X_test.npy", X_test_processed)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data preprocessing complete. Processed files saved.")




