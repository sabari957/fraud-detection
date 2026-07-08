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

```
F D/
│
├── app.py                  # Main Streamlit dashboard application
├── random_forest.joblib    # Trained Random Forest classifier (scikit-learn 1.6.1)
├── features_5.joblib       # List of 5 selected feature names
├── top_5_fraud.csv         # Historical transaction dataset (284,807 rows)
│
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation (this file)
├── .gitignore              # Git ignore rules
└── REPORT.md               # Technical project report
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.10+**
- pip

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd "F D"
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
python -m streamlit run app.py
```
> **Note for Windows users:** If `streamlit` is not recognized as a command, always use `python -m streamlit run app.py` instead.

Open your browser at **http://localhost:8501**

---

## 🔑 Demo Credentials

| Field | Value |
|---|---|
| Username | `analyst` |
| Password | `password` (any text accepted) |
| Role | Select from dropdown |

### Role Permissions

| Role | Capabilities |
|---|---|
| **Fraud Analyst** | Review alerts, update cases, read-only rules |
| **Data Scientist** | Model diagnostics, feature sandbox, backtesting |
| **Auditor / Compliance** | Read-only view of all cases and reports |
| **System Administrator** | Full access to all modules |

---

## 🧠 Model Information

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Framework | scikit-learn 1.6.1 |
| Dataset | Kaggle Credit Card Fraud Detection (284,807 transactions) |
| Fraud Cases | 492 (0.173% of dataset) |
| Selected Features | `V14`, `V12`, `V10`, `V17`, `V3` |
| Hybrid Scoring | `score = 0.6 × ML_probability + 0.4 × rule_match` |

### Feature Descriptions

| Feature | Role | Description |
|---|---|---|
| **V14** | Primary | Most important PCA component — strongly correlated with fraud patterns |
| **V12** | Secondary | Second-most important; captures unusual transaction routing |
| **V10** | Third | Anomaly component; identifies behavioral deviations |
| **V17** | Fourth | Geo-signature component; flags location anomalies |
| **V3** | Fifth | High-frequency component; detects burst transaction activity |

---

## 🏗️ Architecture

```
Transaction Input (CSV stream)
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│  Random Forest    │     │  Rule-Based Engine   │
│  ML Classifier    │     │  (Custom Conditions) │
│  (5 PCA features) │     │  e.g. v14 < -6.0     │
└────────┬──────────┘     └──────────┬───────────┘
         │  ML Score (0–1)           │  Rule Match (0 or 1)
         └──────────┬────────────────┘
                    ▼
         Hybrid Risk Score:
         0.6 × ML_prob + 0.4 × rule_match
                    │
                    ▼
         ┌──────────────────┐
         │  Alert Queue &   │
         │  Case Management │
         └──────────────────┘
```

---

## 📊 Validation Results

| Check | Result |
|---|---|
| Python syntax | ✅ Clean |
| All imports resolve | ✅ Passed |
| Model loads correctly | ✅ RandomForestClassifier |
| Feature alignment | ✅ `['v14', 'v12', 'v10', 'v17', 'v3']` |
| CSV loads (284,807 rows) | ✅ Passed |
| Test fraud prediction | ✅ `0.9979` (fraud row correctly scored) |
| Streamlit server starts | ✅ Running on `localhost:8501` |

---

## 🛠️ Tech Stack

- **Python 3.14** — Core language
- **Streamlit 1.58** — Dashboard framework
- **Plotly 6.8** — Interactive visualizations (ROC, scatter, gauge, heatmap)
- **pandas 3.0** — Data manipulation
- **NumPy 2.4** — Numerical operations
- **scikit-learn 1.6.1** — Random Forest model
- **joblib 1.5** — Model serialization

---

## 📄 License

This project is developed for internship and educational purposes.

---

*GuardianEye — Hybrid AI & Rule-Based Fraud Detection Suite*
