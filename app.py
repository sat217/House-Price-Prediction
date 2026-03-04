import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------
# Page Configuration (must be first)
# -----------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

# -----------------------------------
# Paths
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")


# -----------------------------------
# Load model + template
# -----------------------------------
@st.cache_resource
def load_setup():
    try:
        model = joblib.load(MODEL_PATH)

        df_template = (
            pd.read_csv(DATA_PATH)
            .iloc[[0]]
            .drop(columns=["SalePrice", "Id"])
        )

        return model, df_template

    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.stop()


# -----------------------------------
# Main App
# -----------------------------------
def main():

    st.title("🏠 House Price Prediction App")

    st.write(
        "Enter a few property features below and the model will estimate the house price."
    )

    model, df_template = load_setup()

    st.header("Property Features")

    col1, col2 = st.columns(2)

    with col1:
        overall_qual = st.number_input(
            "Overall Quality (1-10)",
            min_value=1,
            max_value=10,
            value=5
        )

        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=100,
            max_value=10000,
            value=1500
        )

        garage_cars = st.number_input(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=10,
            value=2
        )

    with col2:
        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=1000
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=2000
        )

    st.markdown("---")

    if st.button("Predict House Price"):

        input_data = df_template.copy()

        input_data["OverallQual"] = overall_qual
        input_data["GrLivArea"] = gr_liv_area
        input_data["GarageCars"] = garage_cars
        input_data["TotalBsmtSF"] = total_bsmt_sf
        input_data["YearBuilt"] = year_built

        try:

            with st.spinner("Predicting price..."):

                prediction = model.predict(input_data)[0]
                prediction = float(prediction)

            st.success(f"💰 Estimated House Price: **${prediction:,.2f}**")

            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -----------------------------------
# Run app
# -----------------------------------
if __name__ == "__main__":
    main()