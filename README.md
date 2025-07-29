
# 🕵️ Botnet Detection App

This project is a **Botnet Detection Web Application** developed using **Streamlit** and **LightGBM**, focused on identifying malicious network traffic patterns based on synthetically generated and labeled data.

## 🚀 Overview

The app leverages network traffic features such as login attempts, validation failures, flow sizes, and more, to determine whether the traffic is **normal** or **botnet-infected**.

## 🔍 Features

- 📤 Upload or enter network traffic data for prediction
- ⚡ Real-time botnet traffic classification using a trained LightGBM model
- 📈 Visualizations including ROC Curve, Confusion Matrix, and SHAP feature importance
- 🧪 Custom dataset generation for synthetic botnet data
- 🧠 Model training script with evaluation metrics

## 📁 Project Structure

- `app.py`: Streamlit app for user interface and inference
- `model.py`: Trains the LightGBM model and saves it as `model.pkl`
- `botnet_csv.py`: Generates a synthetic labeled dataset for training
- `botnetDetection.py`: Logistic regression benchmark and ROC analysis
- `requirements.txt`: Lists all dependencies

## 📊 Dataset

The dataset is synthetically generated using rule-based logic via `botnet_csv.py`. It consists of 5000 records with balanced classes of normal and botnet traffic, stored in `synthetic_botnet_dataset_large.csv`.

### Sample Features Used:

- login_Attempts
- failed_Validations
- avg_flow
- current_flow
- packets_flow
- flow_diff
- failed_login_ratio
- flow_per_packet
- login_flow_ratio
- validation_efficiency

## 🧠 Model

- **Algorithm**: LightGBM Classifier (with `class_weight='balanced'`)
- **Scaler**: StandardScaler (for consistency in app and training)
- **Evaluation**: AUC, Confusion Matrix, ROC Curve

## 📦 Installation

```bash
git clone https://github.com/Nishath8/Botnet-Detection-App.git
cd Botnet-Detection-App
pip install -r requirements.txt
```

## ▶️ Running the App

```bash
streamlit run app.py
```

Then go to `http://localhost:8501` in your browser.


## 📌 Future Enhancements

- Support for real-world datasets like CICIDS2017 or UNSW-NB15
- Improved visualizations and alerts
- Model comparison: LightGBM vs Logistic Regression vs Random Forest



*Developed with ❤️ by Nishath8 | Last updated: July 29, 2025*
