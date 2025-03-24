import requests
import pandas as pd
import time

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Remove target column
df = df.drop(columns=["Class"])

# Send transactions in real-time
for i in range(50):  # Simulate 50 transactions
    data = df.iloc[i].to_dict()
    
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    result = response.json()
    
    print(f"Transaction {i+1}: Fraud={result['fraud']}")
    
    time.sleep(2)  # Wait 2 seconds before sending next transaction
