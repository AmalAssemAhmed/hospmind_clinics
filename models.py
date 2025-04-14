import joblib
import tensorflow as tf
import streamlit as st
import os
import requests
#load ECG model
@st.cache_resource
def load_ecg_model():
    model_path = "ecg_model.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/hospiminds_models/resolve/main/my_ecg_model.h5"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

#load chest X ray model
@st.cache_resource
def load_chest_model():
    model_path = "chest_mdel.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/hospiminds_models/resolve/main/chest_efficientb0_h5.h5"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

#load MRI Brain tumor ray model
@st.cache_resource
def load_MRI_model():
    model_path = "MRI_mdel.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/hospiminds_models/resolve/main/brain_tumor.h5"
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
