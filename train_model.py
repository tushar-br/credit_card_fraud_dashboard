import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Features & Target
X = df.drop(columns=["Class"])  # Drop target column
y = df["Class"]  # Target (Fraud = 1, Not Fraud = 0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "model/fraud_model.pkl")
print("Model saved as fraud_model.pkl")
