# 🛡️ GuardianEye — Real-Time Fraud Detection Dashboard

> **Internship Project** | Machine Learning · Fraud Detection · Interactive Dashboard  
> Built with Python, Streamlit, Plotly, and scikit-learn Random Forest

---

## 📌 Project Overview

**GuardianEye** is an internship-ready, end-to-end interactive dashboard for a **Real-Time Scalable Fraud Detection System**. It combines a trained **Random Forest classifier** with a custom **rule-based engine** into a hybrid scoring pipeline — simulating how enterprise fraud operations centers work in practice.

The dashboard ingests transactions in real-time, scores them using 5 PCA-extracted features (`V3`, `V10`, `V12`, `V14`, `V17`), assigns hybrid risk scores, and provides full investigation, reporting, and model governance tooling.

---

## 🚀 Features

| Module | Description |
|---|---|
| 🔐 **Login & RBAC** | Role-based authentication (Analyst, Auditor, Data Scientist, Admin) |
| 🌐 **Executive Overview** | KPI cards, alert timeline, 3D PCA scatter visualization |
| ⚡ **Live Stream Monitor** | Real-time transaction ingestion with ML + rule hybrid scoring |
| 🔎 **Case Investigation** | Forensic workspace with SHAP-style feature attribution charts |
| ⚙️ **Dynamic Rule Builder** | Write, backtest, and deploy custom logical fraud detection rules |
| 🧠 **Model Governance** | ROC/PR curves, confusion matrix, feature importances, prediction sandbox |
| 📊 **Reporting & Exports** | One-click CSV/JSON audit trail export + webhook config simulation |

---

## 🗂️ Project Structure
fraud-detection/
│
├── app.py                  # Main Streamlit dashboard application
├── random_forest.joblib    # Trained Random Forest classifier
├── features_5.joblib       # List of 5 selected feature names
├── top_5_fraud.csv         # Historical transaction dataset
│
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
└── REPORT.md               # Technical project report

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/sabari957/fraud-detection.git
cd fraud-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python -m streamlit run app.py

🧠 Model InformationAlgorithm: Random Forest | Dataset: Kaggle Credit Card Fraud | Features: V3, V10, V12, V14, V17
GuardianEye — Hybrid AI & Rule-Based Fraud Detection Suite

### **Now commit and push**
```bash
git add README.md
git commit -m "docs: update README with full project details"
git push origin main
