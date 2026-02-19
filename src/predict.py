import pickle
from preprocess import clean_text


def load_model(model_path):
    with open(model_path, "rb") as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model


def predict_news(text, vectorizer, model):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    label = "REAL" if pred == 1 else "FAKE"
    return label, round(prob * 100, 2)
