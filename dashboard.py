import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import requests
import threading
import time

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Credit Card Fraud Detection - Live Dashboard"

# Global transaction storage
transactions = []

# Layout
app.layout = html.Div([
    html.H1("Real-Time Credit Card Fraud Detection"),
    dcc.Interval(id="interval-component", interval=3000, n_intervals=0),  # Refresh every 3 sec
    html.Div(id="live-update-table")
])

# Fetch transactions in a separate thread
def fetch_transactions():
    df = pd.read_csv("dataset/creditcard.csv")
    df = df.drop(columns=["Class"])  # Remove target column

    for i in range(50):  # Simulate 50 transactions
        data = df.iloc[i].to_dict()
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        result = response.json()
        
        transactions.append({
            "ID": i+1,
            "Fraud": "YES" if result["fraud"] == 1 else "NO"
        })
        
        time.sleep(2)  # Wait 2 seconds before next transaction

# Start transaction thread
threading.Thread(target=fetch_transactions, daemon=True).start()

# Update table dynamically
@app.callback(Output("live-update-table", "children"), [Input("interval-component", "n_intervals")])
def update_table(n):
    df = pd.DataFrame(transactions)

    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(len(df))])
    ])

# Run the dashboard
if __name__ == "__main__":
    app.run_server(debug=True)
