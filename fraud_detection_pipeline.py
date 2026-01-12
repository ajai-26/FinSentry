import pandas as pd
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# --- CONFIGURATION ---
DATASET_PATH = "creditcard.csv"
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_order.pkl"

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- MODEL TRAINING FUNCTION ---
def train_model():
    """
    Loads data, preprocesses it, trains an XGBoost model,
    and saves the model, scaler, and feature order to disk.
    """
    print("--- 1. Starting Model Training ---")

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'. Please download it and place it in the same folder.")
        return

    print("Dataset found. Loading data...")
    df = pd.read_csv(DATASET_PATH)

    # Use all features except for 'Class' (the target) and 'Time'
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    # Save the exact order of columns used for training. This is crucial for prediction.
    joblib.dump(list(X.columns), FEATURES_PATH)
    print("Feature order saved.")

    # Split data into training and testing sets, ensuring class proportions are maintained
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training and testing sets.")

    # Scale the features and save the scaler object
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print("Data scaled and scaler saved.")

    print("Applying SMOTE to balance the training data... (This may take a moment)")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("SMOTE applied successfully.")

    print("Training XGBoost model...")
    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=200,      # Number of decision trees
        max_depth=5,           # Maximum depth of each tree
        learning_rate=0.1,     # Step size for each iteration
        n_jobs=-1,             # Use all available CPU cores
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(model, MODEL_PATH)
    print("Model training complete and model saved.")

    # Evaluate the model on the unseen test data
    print("\n--- Model Performance on Test Set ---")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))
    
    print("\n✅ --- Training phase complete. Your model is ready! ---")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    train_model()

