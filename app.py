import streamlit as st
import pandas as pd
import joblib
import os

# Paths configuration
MODEL_PATH = os.path.join("models", "house_price_model.pkl")
DATA_PATH = os.path.join("data", "train.csv")

@st.cache_resource
def load_setup():
    """Load the trained model and a base feature template for robust prediction."""
    model = joblib.load(MODEL_PATH)
    # Extract structural baseline/shape of features for the complex ML pipeline
    # We grab the first row from train.csv to hold dummy valid values for all the other 74 missing features.
    df_template = pd.read_csv(DATA_PATH).iloc[[0]].drop(columns=['SalePrice', 'Id'])
    return model, df_template

def main():
    st.set_page_config(page_title="House Price Predictor", layout="centered")
    st.title("House Price Prediction App")
    st.write("Enter house details below to estimate its expected sale price.")

    model, df_template = load_setup()

    st.header("Property Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5, step=1)
        gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", min_value=100, max_value=10000, value=1500, step=50)
        garage_cars = st.number_input("Garage Cars Capacity (0-10)", min_value=0, max_value=10, value=2, step=1)
        
    with col2:
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=1000, step=50)
        year_built = st.number_input("Original Construction Year", min_value=1800, max_value=2026, value=2000, step=1)

    st.markdown("---")
    
    if st.button("Predict"):
        # Copy template structural format and plug in the target inputs exactly
        input_data = df_template.copy()
        
        input_data["OverallQual"] = overall_qual
        input_data["GrLivArea"] = gr_liv_area
        input_data["GarageCars"] = garage_cars
        input_data["TotalBsmtSF"] = total_bsmt_sf
        input_data["YearBuilt"] = year_built

        # Make prediction
        try:
            with st.spinner("Calculating..."):
                prediction = model.predict(input_data)[0]
            st.success(f"Estimated House Price: **${prediction:,.2f}**")
            st.balloons()
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
