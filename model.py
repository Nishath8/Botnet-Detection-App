import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from lightgbm import LGBMClassifier

# --- Load dataset ---
df = pd.read_csv("synthetic_botnet_dataset_large.csv")  # Update path if needed

# --- Base Feature Engineering ---
df["flow_diff"] = df["current_flow"] - df["avg_flow"]
df["failed_login_ratio"] = df["failed_Validations"] / (df["login_Attempts"] + 1e-5)
df["flow_per_packet"] = df["current_flow"] / (df["packets_flow"] + 1e-5)

# --- NEW Feature Engineering ---
df["login_flow_ratio"] = df["login_Attempts"] / (df["packets_flow"] + 1e-5)
df["validation_efficiency"] = df["failed_Validations"] / (df["current_flow"] + 1e-5)

# --- Feature Selection ---
features = [
    "login_Attempts", "failed_Validations", "avg_flow", "current_flow", "packets_flow",
    "flow_diff", "failed_login_ratio", "flow_per_packet",
    "login_flow_ratio", "validation_efficiency"
]

X = df[features]
y = df["botnet"]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- Optional: Scaling (LightGBM doesn’t require it, but we save for consistency with app.py) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- LightGBM Model Training ---
model = LGBMClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# --- Predictions ---
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# --- Evaluation ---
print("✅ Model Evaluation:")
print("Accuracy:", model.score(X_test_scaled, y_test))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save model and scaler ---
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ model.pkl and scaler.pkl saved successfully (LightGBM version).")
