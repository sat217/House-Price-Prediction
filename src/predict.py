import os
import pandas as pd
import joblib

def main():
    print("Loading test data...")
    test_path = os.path.join("..", "data", "test.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join("data", "test.csv")
        
    df_test = pd.read_csv(test_path)
    test_ids = df_test['Id']
    X_test = df_test.drop(columns=['Id'])

    # Load the trained model
    model_path = os.path.join("..", "models", "house_price_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "house_price_model.pkl")

    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)

    print("Generating predictions...")
    preds = pipeline.predict(X_test)

    # Generate submission file format
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': preds
    })

    output_path = os.path.join("..", "submission.csv")
    # if running from project root:
    if os.path.exists("data"):
        output_path = "submission.csv"

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
