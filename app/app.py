import streamlit as st
import sys
import os

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")

sys.path.append(SRC_PATH)

from predict import load_model, predict_news

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🧠",
    layout="wide"
)

# ---------- PREMIUM DARK + GLASS CSS ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Glass card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* Title gradient */
.title {
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 18px;
}

/* Result styles */
.real {
    color: #22c55e;
    font-size: 26px;
    font-weight: bold;
}

.fake {
    color: #ef4444;
    font-size: 26px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🧠 AI Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time NLP powered news authenticity analyzer</div>', unsafe_allow_html=True)

st.write("")

# ---------- LOAD MODEL ----------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Run: python train_model.py")
    st.stop()

vectorizer, model = load_model(MODEL_PATH)

# ---------- LAYOUT ----------
left, right = st.columns([2, 1])

# ---------- INPUT PANEL ----------
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📰 Enter News Text")

    user_input = st.text_area(
        "Paste headline or full article:",
        height=220,
        placeholder="Type or paste news content here..."
    )

    predict_btn = st.button("🔍 Analyze News", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- INFO PANEL ----------
with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("⚙️ Model Details")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Vectorizer:** TF-IDF")
    st.write("**Task:** Binary Classification")
    st.write("**Framework:** Scikit-learn + Streamlit")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION ----------
if predict_btn:
    if user_input.strip() == "":
        st.warning("⚠ Please enter news text.")
    else:
        label, confidence = predict_news(user_input, vectorizer, model)

        st.write("")
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Result")

        if label == "REAL":
            st.markdown(f'<div class="real">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fake">🚨 FAKE NEWS</div>', unsafe_allow_html=True)

        st.write(f"**Confidence Score:** {confidence}%")
        st.progress(int(confidence))

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="footer">Built by Avinash Punjare • B.Tech AI • ML Project</div>', unsafe_allow_html=True)
