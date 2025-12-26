import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# Load saved artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("transformer.pkl", "rb") as f:
        transformer = pickle.load(f)

    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, transformer, columns


model, transformer, columns = load_artifacts()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("🔍 Machine Learning Prediction App")
st.write("Enter feature values to get prediction")

# -------------------------------
# User Input Form
# -------------------------------
input_data = {}

st.subheader("Input Features")

for col in columns:
    input_data[col] = st.number_input(
        label=f"{col}",
        value=0.0,
        step=0.1
    )

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    try:
        # Apply preprocessing
        transformed_input = transformer.transform(input_df)

        # Make prediction
        prediction = model.predict(transformed_input)

        st.success("Prediction Successful ✅")
        st.metric(label="Predicted Value", value=round(float(prediction[0]), 4))

    except Exception as e:
        st.error("Prediction Failed ❌")
        st.exception(e)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit • Scikit-learn")
