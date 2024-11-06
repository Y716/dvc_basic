# scripts/train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()

    model = train_model(X_train, y_train)
    
    # Save the trained model
    dump(model, "models/train_model/model.joblib")
