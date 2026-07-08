# Changelog

All notable changes to this project will be documented in this file.

---

## [1.0.0] — 2026-07-05

### Added
- Full Streamlit dashboard (`app.py`) with 6 interactive modules
- Role-Based Access Control (RBAC) with 4 user roles:
  - Fraud Analyst, Auditor/Compliance, Data Scientist, System Administrator
- **Executive Overview** module:
  - 4 dynamic KPI cards (total alerts, critical risk count, avg risk index, open cases)
  - Alert timeline with rolling average trend line
  - Case resolution status donut chart
  - 3D PCA feature scatter (V3, V14, V17) with color-mapped risk scores
- **Live Stream Monitor** module:
  - Real-time transaction ingestion simulation from `top_5_fraud.csv`
  - ML inference via Random Forest on each tick
  - Dynamic rule evaluation per transaction
  - Hybrid risk score calculation (60% ML + 40% rule)
  - Color-coded severity log feed (Critical / Warning / Info)
  - Configurable tick speed and alert threshold sliders
- **Case Investigation** module:
  - Incident Registry Queue with severity and status filters
  - Forensic Analysis Workspace per alert:
    - Case metadata (timestamp, hybrid score, assignee, status)
    - SHAP-approximated per-feature attribution bar chart
    - Ground truth verification badge
  - Action panel: update status, assign analyst, add audit notes
  - RBAC enforcement: Auditor role is read-only
- **Dynamic Rule Builder** module:
  - View, toggle, and delete deployed rules
  - Two pre-seeded rules (V14 < -6.0 and V12+V17 multi-factor)
  - Rule creator form with Python boolean expression syntax
  - Backtest engine: evaluates rule over 284,807 historical transactions
  - Reports TP, FP, Precision, Recall instantly via `pandas.eval()`
  - Save & deploy directly into the live stream engine
- **Model Governance & Explainability** module:
  - Performance Diagnostics tab:
    - ROC Curve with AUC (stratified 20k-sample validation)
    - Precision-Recall Curve with AUC
    - Dynamic Confusion Matrix heatmap
    - Global Gini Feature Importance bar chart
  - Feature Sandbox Simulator tab:
    - 5 interactive sliders (V3, V10, V12, V14, V17)
    - Real-time fraud risk gauge indicator
    - Dynamic risk level badge (Secure / Moderate / High Fraud Risk)
- **Reporting & Exports** module:
  - Full audit table with case metadata
  - One-click CSV export with timestamped filename
  - One-click JSON export with timestamped filename
  - Webhook/PagerDuty/Slack integration configuration simulation
- Model evaluation script (`evaluate_model.py`)
- Project documentation files:
  - `README.md` — Setup guide and project overview
  - `REPORT.md` — Technical internship report
  - `requirements.txt` — Pinned dependency versions
  - `LICENSE` — MIT License with dataset attribution
  - `.gitignore` — Standard Python gitignore
  - `.streamlit/config.toml` — Streamlit dark theme and server config
  - `CHANGELOG.md` — This file

### Model Performance (verified)
- ROC-AUC: **0.9870**
- Precision: **100.00%**
- Recall: **83.74%**
- F1 Score: **0.9115**
- Overall Accuracy: **99.61%**
