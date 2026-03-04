# House Price Prediction

## Project Goal
This project aims to predict house prices using the Kaggle House Prices dataset. The goal is to build a complete machine learning pipeline, from exploratory data analysis (EDA) and preprocessing to model training and evaluation, ultimately predicting the SalePrice of various houses.

## Dataset
The dataset utilized is the standard Kaggle House Prices dataset. 
- `train.csv`: Used for training and validating the model.
- `test.csv`: Used to generate final predictions.
- `sample_submission.csv`: Example of the output prediction format.

## Preprocessing Steps
1. **Missing Data Handling**:
   - Numerical features: Missing values imputed using the median.
   - Categorical features: Missing values imputed using the most frequent value (mode).
2. **Encoding**:
   - Categorical variables are encoded using `OneHotEncoder`.
3. **Scaling**:
   - Numerical variables are standardized using `StandardScaler`.
4. **Pipeline**:
   - All steps are encapsulated in a `scikit-learn` `ColumnTransformer` for reproducibility and clean code.

## Models Used
Three regression models were evaluated:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

The performance is measured using Root Mean Squared Error (RMSE) and R² Score on a 20% validation split. The best performing model on the validation split is chosen to be trained on the full dataset.

## Evaluation Results
- **Linear Regression**: 
  - RMSE: 29475.52
  - R² Score: 0.8867
- **Random Forest**:
  - RMSE: 28432.11
  - R² Score: 0.8946
- **Gradient Boosting**: 
  - RMSE: 26316.14
  - R² Score: 0.9097

**Best Model Selected**: Gradient Boosting (RMSE: 26316.14, R²: 0.9097)

## How to Run the Project

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate EDA Notebook**:
   ```bash
   cd notebooks
   python generate_notebook.py
   ```
   This will create `house_price_prediction.ipynb` which you can then explore using Jupyter.

3. **Train the Model**:
   ```bash
   python src/train_model.py
   ```
   This script will pre-process data, evaluate all three models, select the best one, train it on the full training set, and save the binary model to `models/house_price_model.pkl`.

4. **Generate Predictions**:
   ```bash
   python src/predict.py
   ```
   This will output a `submission.csv` containing the `Id` and `SalePrice` predictions for the test set.

## Local Testing Instructions

To test the project and web interface locally, follow these steps:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train_model.py
   ```

3. Generate test set predictions:
   ```bash
   python src/predict.py
   ```

4. Run the web interface:
   ```bash
   streamlit run app.py
   ```

