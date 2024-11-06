# scripts/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def download_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    X = df.drop(columns=["species"])
    y = df["species"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Save the processed data
    pd.DataFrame(X_train).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)
