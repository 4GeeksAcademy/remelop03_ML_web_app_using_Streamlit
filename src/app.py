from pickle import load
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import numpy as np

model = load(open("../models/logistic_regression_model.sav", "rb"))

st.set_page_config(page_title="Diabetes prediction", page_icon="🧊", layout="wide")

with st.container():
    st.title("Diabetes prediction APP")
    st.write("The developed model help us to predict, based on diagnostic measures, whether or not a patient has diabetes.")

with st.container():
    st.subheader("Enter the patient's diagnostic measures")
    col1 = st.columns(1)[0]
    with col1:
        Age=st.slider("Age. Age of patient (numeric)", 18, 100, 18)
        Pregnancies=st.slider("Pregnancies. Number of pregnancies of the patient (numeric)", 0, 20, 0)
        Glucose=st.slider("Plasma glucose concentration 2 hours after an oral glucose tolerance test (numeric)", 0, 200, 0)
        SkinThickness=st.slider("Triceps skin fold thickness (measured in mm) (numeric)", 0, 100, 0)
        Insulin=st.slider("2-hour serum insulin (measured in mu U/ml) (numeric)", 0, 900, 0)
        BMI=st.slider("Body mass index (numeric)", 0, 100, 0)
        DiabetesPedigreeFunction=st.slider("Diabetes Pedigree Function (numeric)", 0.0, 3.0, 0.0, 0.001)

with st.container():
    st.markdown("---")
    if st.button("Diabetes Predictor"):
        input_data = np.array([Age, Pregnancies, Glucose, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction])
        input_data = input_data.reshape(1,-1)
        prediction = model.predict(input_data)[0]

        if prediction==1:
            st.success(f"Diabetes")
        else:
            st.success(f"No Diabetes")

st.markdown("---")
