"""
Model Training & Evaluation — Titanic Survival Prediction
-----------------------------------------------------------
Trains Logistic Regression and Random Forest, compares them with
proper metrics (not just accuracy), and saves the better model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

os.makedirs("models", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ---------------------------------------------------------------
# 1. Load model-ready data
# ---------------------------------------------------------------
df = pd.read_csv("data/titanic_model_ready.csv")
X = df.drop(columns=["survived"])
y = df["survived"]

FEATURE_COLUMNS = X.columns.tolist()
with open("models/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)

# Stratified split keeps the survival ratio consistent in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scale features (helps Logistic Regression converge + matters for coefficients)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# ---------------------------------------------------------------
# 2. Train models
# ---------------------------------------------------------------
results = {}

# --- Logistic Regression (baseline, interpretable) ---
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, y_pred_lr),
    "precision": precision_score(y_test, y_pred_lr),
    "recall": recall_score(y_test, y_pred_lr),
    "f1": f1_score(y_test, y_pred_lr),
    "roc_auc": roc_auc_score(y_test, y_proba_lr),
}

# --- Random Forest (with light grid search for max_depth / n_estimators) ---
rf_base = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 2, 4],
}
grid = GridSearchCV(rf_base, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)  # Random Forest doesn't need scaling
rf = grid.best_estimator_
print("Best RF params:", grid.best_params_)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, y_pred_rf),
    "precision": precision_score(y_test, y_pred_rf),
    "recall": recall_score(y_test, y_pred_rf),
    "f1": f1_score(y_test, y_pred_rf),
    "roc_auc": roc_auc_score(y_test, y_proba_rf),
}

# 5-fold cross-validation (F1) to check stability, not just a single split
cv_scores_lr = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring="f1")
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
results["Logistic Regression"]["cv_f1_mean"] = cv_scores_lr.mean()
results["Logistic Regression"]["cv_f1_std"] = cv_scores_lr.std()
results["Random Forest"]["cv_f1_mean"] = cv_scores_rf.mean()
results["Random Forest"]["cv_f1_std"] = cv_scores_rf.std()

print("\n=== Model Comparison ===")
results_df = pd.DataFrame(results).T
print(results_df.round(4))
results_df.to_csv("models/model_comparison.csv")

# ---------------------------------------------------------------
# 3. Pick the winner (by F1 — balances precision/recall, better
#    than accuracy alone since classes aren't perfectly balanced)
# ---------------------------------------------------------------
winner_name = results_df["f1"].idxmax()
winner_model = rf if winner_name == "Random Forest" else log_reg
print(f"\nSelected model: {winner_name}")

if winner_name == "Random Forest":
    y_pred_final = y_pred_rf
    y_proba_final = y_proba_rf
    joblib.dump(rf, "models/titanic_model.pkl")
    needs_scaling = False
else:
    y_pred_final = y_pred_lr
    y_proba_final = y_proba_lr
    joblib.dump(log_reg, "models/titanic_model.pkl")
    needs_scaling = True

with open("models/model_meta.json", "w") as f:
    json.dump({"model_name": winner_name, "needs_scaling": needs_scaling}, f)

print("\nClassification Report:\n", classification_report(y_test, y_pred_final))

# ---------------------------------------------------------------
# 4. Diagnostic plots
# ---------------------------------------------------------------
# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title(f"Confusion Matrix — {winner_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=130)
plt.close()

# ROC curve for both models
plt.figure(figsize=(6, 5))
for name, proba in [("Logistic Regression", y_proba_lr), ("Random Forest", y_proba_rf)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("assets/roc_curve.png", dpi=130)
plt.close()

# Feature importance (Random Forest) — useful talking point even if LR wins
plt.figure(figsize=(7, 6))
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
importances.plot(kind="barh")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=130)
plt.close()

print("\nSaved: confusion_matrix.png, roc_curve.png, feature_importance.png")
print("Saved: models/titanic_model.pkl, models/scaler.pkl, models/model_meta.json")
