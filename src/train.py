import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_process


def train_model(fake_path, real_path, model_path):
    df = load_and_process(fake_path, real_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    with open(model_path, "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("Model trained and saved!")
    return X_test_vec, y_test, model
