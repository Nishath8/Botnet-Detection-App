import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os
from datetime import datetime
import shap

# --- Load model and scaler ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🕵️‍♂️ Botnet Traffic Detection (LightGBM)")
st.write("Enter network traffic features below to check for botnet activity.")

# --- Input fields ---
login_Attempts = st.number_input("Login Attempts", min_value=0)
failed_Validations = st.number_input("Failed Validations", min_value=0)
avg_flow = st.number_input("Average Flow (bytes)", min_value=0.0)
current_flow = st.number_input("Current Flow (bytes)", min_value=0.0)
packets_flow = st.number_input("Packets per Flow", min_value=1)

# --- Feature Engineering (same as train_model.py) ---
flow_diff = current_flow - avg_flow
failed_login_ratio = failed_Validations / (login_Attempts + 1e-5)
flow_per_packet = current_flow / (packets_flow + 1e-5)
login_flow_ratio = login_Attempts / (packets_flow + 1e-5)
validation_efficiency = failed_Validations / (current_flow + 1e-5)

# --- Feature vector ---
input_vector = np.array([[login_Attempts, failed_Validations, avg_flow, current_flow, packets_flow,
                          flow_diff, failed_login_ratio, flow_per_packet,
                          login_flow_ratio, validation_efficiency]])

# --- Predict ---
if st.button("Detect Botnet"):
    input_scaled = scaler.transform(input_vector)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Botnet Traffic Detected! (Confidence: {confidence:.2f})")
        st.subheader("🛡️ Prevention Tips for Botnets")
        st.markdown("""
        - **Use a Firewall:** Block malicious incoming and outgoing traffic.
        - **Update All Software Regularly:** Apply patches to OS, routers, and IoT devices.
        - **Use Strong, Unique Passwords:** Especially for network equipment.
        - **Disable Unused Services and Ports:** Reduce attack surface.
        - **Enable Intrusion Detection Systems (IDS):** Monitor unusual behavior.
        - **Educate Users:** Avoid clicking unknown links or downloading shady attachments.
        """)
    else:
        st.success(f"✅ Normal Traffic (Confidence: {1 - confidence:.2f})")

    st.subheader("🔐 General Network Security Tips")
    st.markdown("""
    - **Use a VPN for secure connections.**
    - **Segment your network** to limit lateral movement if breached.
    - **Regularly backup critical data.**
    - **Deploy endpoint protection tools.**
    - **Log and monitor all activity.**
    - **Use multi-factor authentication (MFA)** wherever possible.
    """)

    # --- Real-Time Logging ---
    log_columns = [
        "login_Attempts", "failed_Validations", "avg_flow", "current_flow", "packets_flow",
        "flow_diff", "failed_login_ratio", "flow_per_packet",
        "login_flow_ratio", "validation_efficiency",
        "prediction", "confidence", "timestamp"
    ]
    log_data = pd.DataFrame([[
        login_Attempts, failed_Validations, avg_flow, current_flow, packets_flow,
        flow_diff, failed_login_ratio, flow_per_packet,
        login_flow_ratio, validation_efficiency,
        prediction, confidence, datetime.now()
    ]], columns=log_columns)

    log_data.to_csv("predictions_log.csv", mode='a', header=not os.path.exists("predictions_log.csv"), index=False)

    # --- ROC Curve ---
    df = pd.read_csv("synthetic_botnet_dataset_large.csv")
    df["flow_diff"] = df["current_flow"] - df["avg_flow"]
    df["failed_login_ratio"] = df["failed_Validations"] / (df["login_Attempts"] + 1e-5)
    df["flow_per_packet"] = df["current_flow"] / (df["packets_flow"] + 1e-5)
    df["login_flow_ratio"] = df["login_Attempts"] / (df["packets_flow"] + 1e-5)
    df["validation_efficiency"] = df["failed_Validations"] / (df["current_flow"] + 1e-5)

    features = [
        "login_Attempts", "failed_Validations", "avg_flow", "current_flow", "packets_flow",
        "flow_diff", "failed_login_ratio", "flow_per_packet",
        "login_flow_ratio", "validation_efficiency"
    ]

    X = df[features]
    y = df["botnet"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.subheader("📈 ROC Curve and AUC")
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc='lower right')
    st.pyplot(fig1)

    st.subheader("🎯 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Botnet"])
    disp.plot(ax=ax2)
    st.pyplot(fig2)

    # --- SHAP Feature Importance ---
    st.subheader("🔍 Feature Importance (SHAP Explanation)")
    explainer = shap.Explainer(model)
    shap_values = explainer(shap.sample(X_test_scaled, 100))
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    st.pyplot(fig3)
