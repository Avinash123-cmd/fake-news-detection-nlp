import sys
import os

print("🚀 Training script started...")

sys.path.append(os.path.abspath("src"))

from train import train_model
from evaluate import evaluate_model

FAKE_PATH = "data/Fake.csv"
REAL_PATH = "data/True.csv"
MODEL_PATH = "models/fake_news_model.pkl"

print("Checking dataset paths...")
print("Fake exists:", os.path.exists(FAKE_PATH))
print("Real exists:", os.path.exists(REAL_PATH))

print("Starting training...")

X_test, y_test, model = train_model(FAKE_PATH, REAL_PATH, MODEL_PATH)

print("Training finished. Evaluating...")

evaluate_model(X_test, y_test, model)

print("✅ DONE")
