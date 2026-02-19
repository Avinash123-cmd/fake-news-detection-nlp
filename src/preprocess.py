import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


def load_and_process(fake_path: str, real_path: str) -> pd.DataFrame:
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real], ignore_index=True)

    df["content"] = df["title"] + " " + df["text"]
    df["content"] = df["content"].apply(clean_text)

    return df[["content", "label"]]
