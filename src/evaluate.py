from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(X_test, y_test, model):
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
