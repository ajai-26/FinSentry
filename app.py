import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
import warnings

# --- SETUP ---
# Initialize the Flask web application
app = Flask(__name__)

# Suppress irrelevant warnings for a cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# --- LOAD THE TRAINED MODEL AND SUPPORTING FILES ---
# This block runs only once when the server starts up.
try:
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_order = joblib.load("feature_order.pkl")
    print("Model, scaler, and feature order loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model files not found. Please run 'fraud_detection_pipeline.py' first to create them.")
    model = None # Set to None to handle errors gracefully

# --- WEB PAGE ROUTES ---

# This route serves the main homepage (index.html)
@app.route('/')
def home():
    # Flask looks for this file in a 'templates' folder
    return render_template('index.html')

# This route handles the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model failed to load
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    # Get the JSON data sent from the user's browser (from the HTML form)
    data = request.get_json(force=True)
    
    # Convert the incoming data into a pandas DataFrame with the correct feature order
    transaction_df = pd.DataFrame([data])[feature_order]

    # Scale the data using the loaded scaler
    transaction_scaled = scaler.transform(transaction_df)

    # Use the loaded model to make a prediction
    prediction = model.predict(transaction_scaled)
    probability = model.predict_proba(transaction_scaled)

    # Extract the results
    is_fraud = bool(prediction[0])
    fraud_probability = float(probability[0][1]) # Get the probability of the 'fraud' class

    # Send the results back to the HTML page in JSON format
    return jsonify({
        'isFraud': is_fraud,
        'probability': fraud_probability
    })

# --- RUN THE APPLICATION ---
if __name__ == '__main__':
    print("Starting Flask server... Go to http://127.0.0.1:5000 in your browser.")
    # Starts the local web server. debug=True allows for automatic reloading on code changes.
    app.run(debug=True)

