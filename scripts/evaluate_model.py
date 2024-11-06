# scripts/evaluate_model.py
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import load

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    model = load("models/train_model/model.joblib")

    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    
    # Save the evaluation results
    with open("results/evaluation.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}\n")
