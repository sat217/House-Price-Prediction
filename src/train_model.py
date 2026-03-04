import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from preprocess import get_preprocessor

def evaluate_model(y_true, y_pred, name):
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}\n")
    return rmse

def main():
    print("Loading data...")
    train_path = os.path.join("..", "data", "train.csv")
    if not os.path.exists(train_path):
        train_path = os.path.join("data", "train.csv")
        
    df = pd.read_csv(train_path)

    # Separate features and target
    X = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    preprocessor = get_preprocessor(X_train)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    best_model_name = ""
    best_rmse = float('inf')
    best_pipeline = None

    print("Training and evaluating models...\n")
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        rmse = evaluate_model(y_test, y_pred, name)
        
        # Update best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

    print(f"Best model selected: {best_model_name}")

    # Train best model on the FULL dataset
    print(f"Training {best_model_name} on the full dataset...")
    final_pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor(X)),
        ('regressor', models[best_model_name])
    ])
    final_pipeline.fit(X, y)

    # Save the model
    os.makedirs(os.path.join("..", "models"), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("..", "models", "house_price_model.pkl")
    if os.path.exists("models"):
        model_path = os.path.join("models", "house_price_model.pkl")
        
    joblib.dump(final_pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
