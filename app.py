import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load the trained fraud detection model
model = joblib.load("model/fraud_model.pkl")  # Ensure the correct path

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define a prediction route for 30 features
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Expecting 30 values

        # Make prediction
        prediction = model.predict(features)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)







'''
The values you entered are numerical features used in credit card fraud detection models, but they are not actual credit
 card numbers. Instead, they represent different transaction-related features extracted from a real dataset (like the "V1, V2,
   ..., V30" features in the famous Credit Card Fraud Detection dataset from Kaggle).
'''


# Fraud values 
'''-6.4, 7.8, -5.9, 8.2, -7.1, 6.3, -4.5, 9.0, -5.8, 7.6, 
-3.9, 8.5, -6.1, 7.4, -4.2, 9.1, -5.7, 8.3, -6.9, 7.1, 
-4.3, 9.2, -5.5, 7.9, -3.8, 6.7, -4.9, 8.8, -5.2, 7.5
'''

# Not Fraud values
'''-3.2, 5.1, -2.7, 4.8, -1.2, 3.9, -0.5, 2.3, -3.6, 4.2, 
-1.5, 3.8, -2.4, 5.3, -3.1, 4.9, -0.9, 2.7, -4.0, 3.5, 
-2.2, 4.6, -1.8, 5.0, -3.5, 2.9, -0.7, 3.3, -2.9, 4.1
'''