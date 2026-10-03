"""Optional Streamlit interface for the trained house price model."""
import streamlit as st
from src.predict import predict_one, MODEL_FILES

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.write("Estimate median house value using a model trained on the California Housing dataset.")
st.caption("Educational demonstration only; not a current property valuation service.")

model_name = st.selectbox("Regression model", list(MODEL_FILES))
st.subheader("Enter the area's dataset features")
defaults = {
    "MedInc": 5.0, "HouseAge": 20.0, "AveRooms": 5.0, "AveBedrms": 1.0,
    "Population": 1000.0, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0
}
with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)
    for i, (feature, default) in enumerate(defaults.items()):
        with cols[i % 2]:
            values[feature] = st.number_input(feature, value=float(default))
    submitted = st.form_submit_button("Predict median value")

if submitted:
    try:
        prediction = predict_one(values, model_name)
        st.success(f"Predicted median value: ${prediction * 100000:,.0f}")
        st.caption(f"Dataset target value: {prediction:.3f} (units of $100,000)")
    except FileNotFoundError as exc:
        st.error(str(exc))
