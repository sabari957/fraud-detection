# 🛡️ GuardianEye — Real-Time Fraud Detection Dashboard

> **Internship Project** | Machine Learning · Fraud Detection · Interactive Dashboard  
> Built with Python, Streamlit, Plotly, and scikit-learn Random Forest

---

## 🚀 Project Overview

**GuardianEye** is an internship-ready, end-to-end interactive dashboard for a **Real-Time Scalable Fraud Detection System**. 
It combines a trained **Random Forest classifier** with a custom **rule-based engine** into a hybrid scoring pipeline – simulating how enterprise fraud operations centers work in practice.

This project demonstrates data preprocessing, model training, evaluation, and deployment in a production-like environment.

## ✨ Key Features

- **Real-time Prediction**: Manual input + CSV upload for batch predictions
- **Interactive Dashboard**: Plotly visualizations for fraud distribution, feature importance, and model metrics
- **Hybrid Scoring**: ML Model + Rule-based engine for reduced false positives
- **Model Explainability**: SHAP/feature importance charts included
- **Deployment Ready**: Clean Streamlit UI, one-command run

## 🧠 Model Information

- **Algorithm**: Random Forest Classifier
- **Dataset**: Kaggle Credit Card Fraud Detection Dataset
- **Key Features Used**: V3, V10, V12, V14, V17
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Files**: `random_forest.joblib`, `features_5.joblib`

## 🛠️ Tech Stack

`Python` `Streamlit` `scikit-learn` `Pandas` `NumPy` `Plotly` `joblib`

## 📦 Installation & How to Run

1. Clone the repository
```bash
git clone https://github.com/sabari957/fraud-detection.git
cd fraud-detection

2. Install dependencies
```bash
pip install -r requirements.txt
Run the Streamlit app
bashstreamlit run app.py
Then open http://localhost:8501 in your browser.

📁 Repository Structurejavascriptfraud-detection/
├── app.py                 # Main Streamlit dashboard
├── evaluate_model.py      # Model evaluation script
├── random_forest.joblib   # Trained model
├── features_5.joblib      # Feature scaler/encoder
├── requirements.txt       # Dependencies
├── top_5_fraud.csv        # Sample data
├── README.md              # This file
├── REPORT.md              # Detailed project report
└── WORKFLOW.md            # Project workflow

📊 Results
The Random Forest model achieved high performance on imbalanced fraud data with proper preprocessing and feature selection. 
The dashboard allows fraud analysts to simulate real-time decision making.👨‍💻 Author
SabariInternship Project - Machine Learning & Data Analytics⭐

If you found this project helpful, please star the repo!
