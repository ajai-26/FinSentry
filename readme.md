# FinSentry: AI-Powered Fraud Detection System 🛡️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![ML](https://img.shields.io/badge/Model-Random_Forest-green)
![Framework](https://img.shields.io/badge/Framework-Flask-orange)

**FinSentry** is a machine learning solution designed to detect fraudulent credit card transactions with high precision. It addresses the challenge of **imbalanced data** (where fraud cases are rare) by utilizing advanced preprocessing techniques and a tuned Random Forest classifier to minimize financial risk.

## 🕵️‍♂️ How It Works

1.  **Data Ingestion:** Takes transaction parameters (V1-V28, Amount) as input.
2.  **Preprocessing:** Applies **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset and `StandardScaler` to normalize feature variance.
3.  **Prediction Engine:** Uses a trained **Random Forest Classifier** (`fraud_model.pkl`) to predict the probability of fraud.
4.  **Interface:** A user-friendly Flask web app allows manual transaction verification.

## 🚀 Key Features

* **Precision-Focused:** Optimized to reduce **False Negatives** (missing a real fraud) and **False Positives** (blocking a legitimate user).
* **Real-time Interface:** Web dashboard for instant fraud checks.
* **Scalable Pipeline:** Modular preprocessing pipeline (`fraud_detection_pipeline.py`) capable of handling new data streams.

## 📂 Dataset (Important)

This project relies on the **Credit Card Fraud Detection** dataset.
Due to file size limits, the dataset is **not included** in this repository.

👉 **[Download the Dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)**

*Download `creditcard.csv` and place it in the root directory before retraining the model.*

## 💻 Installation & Run

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/ajai-26/FinSentry.git](https://github.com/ajai-26/FinSentry.git)
    cd FinSentry
    ```

2.  **Install dependencies:**
    *(Ensure you have flask, scikit-learn, pandas, and imbalanced-learn installed)*
    ```bash
    pip install flask scikit-learn pandas imblearn
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```
    Open your browser and go to `http://127.0.0.1:5000/`

## 📊 Project Structure

* `app.py`: Flask application server.
* `fraud_model.pkl`: Pre-trained Random Forest model.
* `scaler.pkl`: Saved scaler object for data normalization.
* `templates/`: HTML files for the web interface.

---
