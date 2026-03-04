
# House Price Prediction

## Project Overview
This project is a production-ready machine learning application for predicting house prices using the Ames Housing dataset. It features a complete ML pipeline and a Streamlit web app for interactive predictions.

## Dataset Description
- **train.csv**: Training data with house features and sale prices.
- **test.csv**: Test data for generating predictions.
- **sample_submission.csv**: Example submission format for Kaggle.

## Machine Learning Pipeline
- **Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features.
- **Model Training**: Trains multiple regression models and selects the best based on validation metrics.
- **Model Saving**: The best model is saved as `models/house_price_model.pkl`.
- **Prediction**: The app uses the trained model to predict house prices based on user input.

## Model Used
- Gradient Boosting Regressor (best performance)
- Other models evaluated: Linear Regression, Random Forest

## Project Structure
```
house-price-prediction/
├── models/
│   └── house_price_model.pkl
├── data/
│   └── train.csv
├── notebooks/
│   └── house_price_prediction.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/sat217/House-Price-Prediction.git
   cd House-Price-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
To launch the Streamlit app:
```bash
python -m streamlit run app.py
```

## Streamlit Cloud Deployment
- Main file: `app.py`
- Branch: `main`
- No path conflicts or dependency issues

## Example Usage
1. Enter house features in the web app
2. Click "Predict House Price"
3. View the estimated price

## License
MIT

