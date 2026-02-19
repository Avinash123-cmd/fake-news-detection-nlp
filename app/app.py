import streamlit as st
import sys
import os

# get absolute path of project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

sys.path.append(SRC_PATH)

from src.predict import load_model, predict_news


st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection using AI")

# load model safely
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please run: python train_model.py")
    st.stop()

vectorizer, model = load_model(MODEL_PATH)

user_input = st.text_area("Enter news headline or content:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_news(user_input, vectorizer, model)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence}%")
