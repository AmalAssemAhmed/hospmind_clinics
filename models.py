import joblib
import tensorflow as tf
import streamlit as st
import os
import requests

@st.cache_resource
def load_ecg_model():
    model_path = "ecg_model.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/ecg_model/resolve/main/my_ecg_model.h5"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_heartattack_model():
    model = joblib.load("heartattack_xgb_best_model.pkl")
    return model

@st.cache_resource
def load_heartattack_scaler():
    scaler = joblib.load("HeartAttack_scaler.pkl")
    return scaler
