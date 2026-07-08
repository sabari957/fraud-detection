# -*- coding: utf-8 -*-
"""
GuardianEye — Fraud Detection Dashboard
========================================
Standalone model evaluation script.

Run this to quickly verify the model's performance metrics
on the full dataset without launching the Streamlit UI.

Usage:
    python evaluate_model.py

Output:
    - Fraud detection rate
    - Precision & Recall at threshold 0.5
    - ROC-AUC score
    - Sample predictions on 5 known fraud rows
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import sys
import warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

print("=" * 60)
print("  GuardianEye — Model Evaluation Script")
print("=" * 60)

# ── Load artifacts ──────────────────────────────────────────────
print("\n[1/4] Loading model and data...")
model = joblib.load("random_forest.joblib")
features = joblib.load("features_5.joblib")
df = pd.read_csv("top_5_fraud.csv")

print(f"  [OK] Model: {type(model).__name__}")
print(f"  [OK] Features: {features}")
print(f"  [OK] Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  [OK] Fraud cases in dataset: {int(df['class'].sum())} ({df['class'].mean():.4%})")

# ── Prepare validation set ──────────────────────────────────────
print("\n[2/4] Preparing stratified validation set...")
df_fraud = df[df["class"] == 1]                              # all 492 fraud
df_normal = df[df["class"] == 0].sample(20000, random_state=42)  # 20k normal
val_df = pd.concat([df_fraud, df_normal]).sample(frac=1, random_state=0)

X_val = val_df[features][model.feature_names_in_]
y_val = val_df["class"]
print(f"  [OK] Validation set size: {len(val_df):,} ({len(df_fraud)} fraud + {len(df_normal):,} normal)")

# ── Run inference ───────────────────────────────────────────────
print("\n[3/4] Running inference...")
y_probs = model.predict_proba(X_val)[:, 1]
y_preds = (y_probs >= 0.5).astype(int)

# ── Report metrics ──────────────────────────────────────────────
print("\n[4/4] Performance Metrics")
print("-" * 40)

roc_auc = roc_auc_score(y_val, y_probs)
precision = precision_score(y_val, y_preds)
recall = recall_score(y_val, y_preds)
f1 = f1_score(y_val, y_preds)
cm = confusion_matrix(y_val, y_preds)

print(f"  ROC-AUC Score   : {roc_auc:.6f}")
print(f"  Precision       : {precision:.4f}  ({precision:.2%})")
print(f"  Recall          : {recall:.4f}  ({recall:.2%})")
print(f"  F1 Score        : {f1:.4f}")

print("\n  Confusion Matrix:")
print(f"  {'':15s}  Predicted Normal  Predicted Fraud")
print(f"  {'Actual Normal':15s}  {cm[0][0]:>16,}  {cm[0][1]:>14,}")
print(f"  {'Actual Fraud':15s}  {cm[1][0]:>16,}  {cm[1][1]:>14,}")

print("\n  Full Classification Report:")
print(classification_report(y_val, y_preds, target_names=["Normal", "Fraud"], digits=4))

# ── Sample predictions ──────────────────────────────────────────
print("-" * 40)
print("  Sample Fraud Case Predictions (known fraud rows):")
sample_fraud = df[df["class"] == 1].head(5)
for i, (_, row) in enumerate(sample_fraud.iterrows()):
    X_s = pd.DataFrame([row[features].values], columns=features)[model.feature_names_in_]
    p = model.predict_proba(X_s)[0][1]
    flag = "[DETECTED]" if p >= 0.5 else "[MISSED]"
    print(f"  Row {i+1}: Fraud Probability = {p:.4f}  → {flag}")

print("\n" + "=" * 60)
print("  Evaluation complete.")
print("=" * 60)
