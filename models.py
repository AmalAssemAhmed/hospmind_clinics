import joblib
import tensorflow as tf
import streamlit as st
import os
import requests

# load ECG model
@st.cache_resource
def load_ecg_model():
    model_path = "ecg_model.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/hospiminds_models/resolve/main/my_ecg_model.h5"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"error loading ECG model: {e}")
            return None
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# load chest X ray model
@st.cache_resource
def load_chest_model():
    model_path = "chest_model.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/hospiminds_models/resolve/main/chest_efficientb0_h5.h5"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"error loadind Chest X-Ray model: {e}")
            return None
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# load MRI Brain tumor model
@st.cache_resource
def load_MRI_model():
    model_path = "MRI_model.h5"
    if not os.path.exists(model_path):
        url = "https://huggingface.co/AmalAssem/hospiminds_models/resolve/main/brain_tumor.h5"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"error loading brain tumor model: {e}")
            return None
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

@st.cache_resource
def load_heartattack_model():
    try:
        model = joblib.load("heartattack_xgb_best_model.pkl")
    except Exception as e:
        st.error(f"error loading Heart Attack model: {e}")
        return None
    return model

@st.cache_resource
def load_heartattack_scaler():
    try:
        scaler = joblib.load("HeartAttack_scaler.pkl")
    except Exception as e:
        st.error(f"error loadig Heart Attack Scaler: {e}")
        return None
    return scale
