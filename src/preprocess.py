import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_data(filepath):
    """Load dataset from CSV."""
    return pd.read_csv(filepath)

def get_preprocessor(X):
    """
    Creates and returns a scikit-learn ColumnTransformer for preprocessing.
    - Numerical: Impute missing with median, scale with StandardScaler
    - Categorical: Impute missing with 'missing', encode with OneHotEncoder
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # We remove 'Id' from numeric features if it's there
    if 'Id' in numeric_features:
        numeric_features.remove('Id')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

if __name__ == "__main__":
    print("Preprocessing module is ready.")
