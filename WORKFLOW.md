# GuardianEye — Dashboard Workflow Overview
### For Project Leader Review

---

## What This Project Does (In Simple Terms)

GuardianEye is an **interactive fraud detection dashboard** built for a bank or fintech company.
It takes credit card transaction data, runs it through a **Machine Learning model (Random Forest)**,
and shows analysts — in real time — which transactions are likely fraudulent.

Think of it like a **CCTV control room**, but for financial transactions.

---

## How to Open the Dashboard

```
Run: python -m streamlit run app.py
Then open: http://localhost:8501
```

---

## Module 1 — Login & Role Selection

**What it does:**
When you open the dashboard, you see a login screen. You enter your username, password, and choose your job role.

**Why it matters:**
Different people in a fraud team have different permissions. For example:
- A **Fraud Analyst** can investigate cases but cannot change rules.
- An **Auditor** can only read data — cannot modify anything.
- A **System Administrator** has full access to everything.

**Workflow:**
```
User opens the app
  → Enters username + password
  → Selects their role (Analyst / Auditor / Data Scientist / Admin)
  → Dashboard unlocks based on that role
```

---

## Module 2 — Executive Overview

**What it does:**
This is the **home screen** for managers. It shows a high-level summary of everything happening in the system right now.

**What you see:**
- How many fraud alerts have been generated today
- How many are critically high-risk (above 70% fraud probability)
- The average risk level across all transactions
- How many cases are still unresolved

Below that, you see charts:
- A **timeline graph** showing risk scores over time
- A **pie chart** showing how many cases are new, under investigation, or resolved
- A **3D scatter plot** showing where suspicious transactions cluster in data space

**Workflow:**
```
Manager logs in
  → Sees KPI numbers at a glance
  → Checks timeline to spot spikes in fraud activity
  → Reviews pie chart to see how well the team is resolving cases
```

---

## Module 3 — Live Stream Monitor

**What it does:**
This simulates **real-time transaction monitoring** — like watching transactions come in from ATMs, online payments, and card swipes, one by one.

**How it works:**
Every few seconds, the system picks a new transaction, runs it through the Random Forest model, checks it against fraud rules, and gives it a **risk score from 0% to 100%**.

A transaction with a score above 75% triggers a **Critical Fraud Alert** (shown in red).

**Workflow:**
```
Analyst clicks "Start Live Stream"
  → System picks a transaction from the dataset
  → Runs it through the AI model → gets fraud probability
  → Checks custom rules (e.g. "Is V14 below -6?")
  → Combines both into a final Hybrid Risk Score
  → Displays result in the live log feed
  → Repeats every 1-3 seconds
Analyst clicks "Pause" to stop
```

---

## Module 4 — Case Investigation

**What it does:**
When a suspicious transaction is flagged, it becomes a **case**. This module is where analysts **open each case, dig into the details, and decide what to do**.

**What you can do:**
- See a list of all flagged transactions (filterable by severity and status)
- Click any case to open a full **forensic workspace**
- See exactly which features (V3, V10, V12, V14, V17) triggered the fraud flag and by how much
- Change the case status: New → In Progress → Confirmed Fraud / False Positive
- Assign the case to a specific analyst
- Write investigation notes for audit records

**Workflow:**
```
Analyst opens Case Investigation
  → Sees list of flagged alerts
  → Clicks on a suspicious one (e.g. ALRT-1042)
  → Reviews the feature chart — V14 is -9.2 (very abnormal)
  → Sees model gave it 94% fraud probability
  → Changes status to "Confirmed Fraud"
  → Assigns it to "Amit Patel (Fraud Specialist)"
  → Writes notes: "V14 severely depleted — confirmed account compromise"
  → Clicks Update
```

---

## Module 5 — Dynamic Rule Builder

**What it does:**
Besides the AI model, fraud teams also use **manual rules** — simple "if this, then flag it" logic.
This module lets analysts **create, test, and deploy** those rules without touching any code.

**Example rules already in the system:**
- If `V14 < -6.0` then flag as suspicious
- If `V12 < -4.0` AND `V17 < -4.0` then flag as high risk

**The Backtest Engine:**
Before deploying a new rule, you can test it against **all 284,807 historical transactions** to see:
- How many frauds it would have caught (True Positives)
- How many innocent transactions it would have wrongly flagged (False Positives)
- Its Precision and Recall scores

**Workflow:**
```
Data Scientist types a new rule: "v10 < -5.0"
  → Clicks "Simulate & Backtest Rule"
  → System tests it against all historical data in seconds
  → Shows results: caught 78 fraud cases, wrongly flagged 12 normal ones
  → Analyst reviews numbers — decides it's good enough
  → Clicks "Save & Deploy" → rule is now active in the live stream
```

---

## Module 6 — Model Governance & Explainability

**What it does:**
This is where **Data Scientists and Compliance Officers** verify that the AI model is working correctly, fairly, and transparently.

**Performance Diagnostics (Tab 1):**
- **ROC Curve** — shows how well the model separates fraud from non-fraud
- **Precision-Recall Curve** — critical for imbalanced datasets like fraud
- **Confusion Matrix** — shows exactly how many fraud cases were caught and missed
- **Feature Importance Chart** — shows which features the model relies on most

**Feature Sandbox (Tab 2):**
- Interactive sliders for each of the 5 features (V3, V10, V12, V14, V17)
- You drag the slider and the fraud risk **gauge updates live**
- This lets anyone understand how the model "thinks" without needing to read code

**Workflow:**
```
Data Scientist opens Model Governance
  → Checks ROC-AUC score (0.987 — excellent)
  → Looks at Confusion Matrix — model caught 412 of 492 fraud cases
  → Opens Feature Sandbox tab
  → Drags V14 slider to -10 → gauge jumps to 98% fraud risk
  → Drags back to 0 → gauge drops to 3% (safe)
  → Confirms the model behaves logically and is explainable
```

---

## Module 7 — Reporting & Exports

**What it does:**
At the end of each shift or investigation cycle, the team needs to **export records** for compliance, audit, and management reporting.

**What you can export:**
- Full case log table (every flagged transaction + analyst notes + resolution)
- Download as **CSV** (for Excel / spreadsheet tools)
- Download as **JSON** (for sending to other systems or APIs)

**Webhook Configuration:**
Simulates connecting to external tools like Slack or PagerDuty so that critical fraud alerts automatically notify the team.

**Workflow:**
```
Compliance Officer opens Reporting & Exports
  → Sees a full table of all cases handled
  → Reviews: Alert ID, Risk Score, ML Confidence, Rules Triggered, Notes
  → Clicks "Download Audit File (CSV)"
  → File saved with today's timestamp
  → Submits to compliance team
```

---

## Summary Table

| Module | Who Uses It | Main Purpose |
|---|---|---|
| Login & RBAC | Everyone | Secure access based on job role |
| Executive Overview | Managers / Directors | High-level fraud activity snapshot |
| Live Stream Monitor | Fraud Analysts | Watch transactions being scored in real time |
| Case Investigation | Fraud Analysts | Investigate, assign, and resolve flagged cases |
| Dynamic Rule Builder | Data Scientists | Create and test custom fraud logic rules |
| Model Governance | Data Scientists / Auditors | Verify model accuracy and explain AI decisions |
| Reporting & Exports | Compliance Officers | Export audit trails and case reports |

---

*GuardianEye — Hybrid AI & Rule-Based Fraud Detection Suite | Internship Project 2026*
