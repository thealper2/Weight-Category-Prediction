import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("rf.pkl", "rb"))

st.title("Weight Category Prediction")
g = st.selectbox("Gender", ["Male", "Female"])
h = st.number_input("Height")
w = st.number_input("Weight")

if st.button("Predict"):
    test = []
    if g == "Male":
        test = np.array([[0, 1, h, w]])

    else:
        test = np.array([[1, 0, h, w]])

    res = model.predict(test)
    print(res)
    st.success("Predicted: " + str(res[0]))
