#!/usr/bin/env python3
"""
============================================================================
ROSHN Lead Conversion Prediction Model
============================================================================
Predicts which active leads are most likely to convert (Won).
Uses the same pipeline approach as the default prediction model.

Features engineered from: source, interactions, response time, lead score,
budget, property type, nationality, time in pipeline, agent performance.

Run: python src/03_train_lead_model.py
============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                             f1_score, accuracy_score, precision_recall_curve, roc_curve)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PATHS
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("ROSHN LEAD CONVERSION PREDICTION MODEL")
print("=" * 70)

leads = pd.read_csv(os.path.join(DATA_DIR, "roshn_leads.csv"))
leads["created_date"] = pd.to_datetime(leads["created_date"])
leads["last_activity_date"] = pd.to_datetime(leads["last_activity_date"])

print(f"\nTotal leads loaded: {len(leads):,}")
print(f"Stage distribution:\n{leads['stage'].value_counts().to_string()}")

# ============================================================================
# 2. DEFINE TARGET
# ============================================================================
# Binary classification: Won (1) vs Not Won (0)
# Exclude currently active leads from training — only train on resolved leads
# Resolved = Won or Lost
# Active leads will be scored after training

resolved_mask = leads["stage"].isin(["Won", "Lost"])
active_mask = ~resolved_mask

leads_resolved = leads[resolved_mask].copy()
leads_active = leads[active_mask].copy()

leads_resolved["target"] = (leads_resolved["stage"] == "Won").astype(int)

print(f"\nResolved leads (training): {len(leads_resolved):,}")
print(f"  Won:  {leads_resolved['target'].sum():,} ({leads_resolved['target'].mean()*100:.1f}%)")
print(f"  Lost: {(1-leads_resolved['target']).sum():,} ({(1-leads_resolved['target'].mean())*100:.1f}%)")
print(f"Active leads (to score):   {len(leads_active):,}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n--- Feature Engineering ---")

def engineer_features(df):
    """Engineer features from lead data."""
    features = pd.DataFrame(index=df.index)
    
    # ---- Numeric features ----
    features["lead_score"] = df["lead_score"]
    features["response_time_hours"] = df["response_time_hours"]
    features["total_interactions"] = df["total_interactions"]
    features["ai_assisted"] = df["ai_assisted"].astype(int)
    
    # ---- Time features ----
    ref_date = df["last_activity_date"].max() if "last_activity_date" in df.columns else pd.Timestamp.now()
    features["days_in_pipeline"] = (df["last_activity_date"] - df["created_date"]).dt.days
    features["days_since_activity"] = (ref_date - df["last_activity_date"]).dt.days
    features["created_month"] = df["created_date"].dt.month
    features["created_quarter"] = df["created_date"].dt.quarter
    features["created_dayofweek"] = df["created_date"].dt.dayofweek
    
    # ---- Response time buckets ----
    features["fast_response"] = (df["response_time_hours"] < 2).astype(int)
    features["slow_response"] = (df["response_time_hours"] > 24).astype(int)
    
    # ---- Interaction intensity ----
    features["interactions_per_day"] = features["total_interactions"] / features["days_in_pipeline"].clip(lower=1)
    
    # ---- Score buckets ----
    features["high_score"] = (df["lead_score"] >= 70).astype(int)
    features["low_score"] = (df["lead_score"] < 40).astype(int)
    
    # ---- Budget as numeric ----
    budget_map = {"500K-1M": 1, "1M-2M": 2, "2M-3M": 3, "3M-5M": 4, "5M-10M": 5, "10M+": 6}
    features["budget_level"] = df["budget_range"].map(budget_map).fillna(2)
    
    # ---- Follow-up ----
    features["follow_up_scheduled"] = df["follow_up_scheduled"].astype(int)
    
    return features

# Engineer features for both sets
features_resolved = engineer_features(leads_resolved)
features_active = engineer_features(leads_active)

# ---- Categorical Encoding ----
cat_cols = ["source", "community_interest", "zone", "property_type_interest",
            "nationality", "budget_range", "assigned_agent"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on all data to handle unseen categories
    all_vals = pd.concat([leads_resolved[col], leads_active[col]]).astype(str)
    le.fit(all_vals)
    features_resolved[col] = le.transform(leads_resolved[col].astype(str))
    features_active[col] = le.transform(leads_active[col].astype(str))
    label_encoders[col] = le

feature_cols = list(features_resolved.columns)
print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# ============================================================================
# 4. PREPARE TRAINING DATA
# ============================================================================
print("\n--- Preparing Training Data ---")

X = features_resolved[feature_cols].values
y = leads_resolved["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

# SMOTE for class balancing (if available)
if HAS_SMOTE:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_bal.shape[0]:,} (class 0: {(y_train_bal==0).sum()}, class 1: {(y_train_bal==1).sum()})")
else:
    X_train_bal, y_train_bal = X_train, y_train
    print(f"SMOTE not available — using class_weight='balanced' instead")

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5. TRAIN MODELS
# ============================================================================
print("\n--- Training Models ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                           class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                                    min_samples_leaf=10, random_state=42),
}

if HAS_XGB:
    scale_pos = (y_train_bal == 0).sum() / max((y_train_bal == 1).sum(), 1)
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1, use_label_encoder=False
    )

results = {}
best_auc = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train_bal)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_bal, y_train_bal)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {"auc": auc, "f1": f1, "accuracy": acc, "model": model,
                     "y_pred_proba": y_pred_proba, "y_pred": y_pred}
    
    print(f"    ROC-AUC: {auc:.4f}  |  F1: {f1:.4f}  |  Accuracy: {acc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model_name = name

print(f"\n  ⭐ Best Model: {best_model_name} (AUC: {best_auc:.4f})")

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================
print("\n--- Cross-Validation (Best Model) ---")
best_model = results[best_model_name]["model"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
if best_model_name == "Logistic Regression":
    cv_scores = cross_val_score(best_model, scaler.transform(X), y, cv=cv, scoring="roc_auc")
else:
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="roc_auc")

print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# 7. SCORE ALL LEADS
# ============================================================================
print("\n--- Scoring All Leads ---")

X_all_resolved = features_resolved[feature_cols].values
X_all_active = features_active[feature_cols].values

if best_model_name == "Logistic Regression":
    proba_resolved = best_model.predict_proba(scaler.transform(X_all_resolved))[:, 1]
    proba_active = best_model.predict_proba(scaler.transform(X_all_active))[:, 1]
else:
    proba_resolved = best_model.predict_proba(X_all_resolved)[:, 1]
    proba_active = best_model.predict_proba(X_all_active)[:, 1]

# Add predictions back
leads_resolved = leads_resolved.copy()
leads_active = leads_active.copy()

leads_resolved["conversion_probability"] = proba_resolved
leads_active["conversion_probability"] = proba_active

# Priority grades
def assign_priority(prob):
    if prob >= 0.75:
        return "Hot"
    elif prob >= 0.50:
        return "Warm"
    elif prob >= 0.25:
        return "Cool"
    else:
        return "Cold"

leads_resolved["lead_priority"] = leads_resolved["conversion_probability"].apply(assign_priority)
leads_active["lead_priority"] = leads_active["conversion_probability"].apply(assign_priority)

# Combine
leads_scored = pd.concat([leads_resolved, leads_active], ignore_index=True)

# Print summary
print(f"\nActive lead priority distribution:")
print(leads_active["lead_priority"].value_counts().to_string())
print(f"\nHot leads (>75% conversion probability): {(leads_active['lead_priority']=='Hot').sum()}")
print(f"Warm leads (50-75%): {(leads_active['lead_priority']=='Warm').sum()}")

# Save scored leads
scored_path = os.path.join(OUTPUT_DIR, "roshn_leads_scored.csv")
leads_scored.to_csv(scored_path, index=False)
print(f"\nScored leads saved to: {scored_path}")

# ============================================================================
# 8. GENERATE CHARTS
# ============================================================================
print("\n--- Generating Evaluation Charts ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("ROSHN Lead Conversion Model — Evaluation Report", fontsize=16, fontweight="bold", y=0.98)
fig.patch.set_facecolor("#FAF8F5")

# 8a. ROC Curves
ax = axes[0, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_pred_proba"])
    ax.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(fontsize=8)
ax.set_facecolor("#FDFCFA")

# 8b. Confusion Matrix (best model)
ax = axes[0, 1]
cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
            xticklabels=["Lost", "Won"], yticklabels=["Lost", "Won"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix ({best_model_name})")

# 8c. Feature Importance (tree model)
ax = axes[0, 2]
if HAS_XGB and "XGBoost" in results:
    imp_model = results["XGBoost"]["model"]
elif "Gradient Boosting" in results:
    imp_model = results["Gradient Boosting"]["model"]
else:
    imp_model = results["Random Forest"]["model"]

importances = pd.Series(imp_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
importances.plot(kind="barh", ax=ax, color="#C9A96E")
ax.set_title("Top 15 Feature Importance")
ax.set_facecolor("#FDFCFA")

# 8d. Probability Distribution
ax = axes[1, 0]
ax.hist(leads_active["conversion_probability"], bins=40, color="#6B8F71", alpha=0.8, edgecolor="white")
ax.axvline(x=0.75, color="#C1666B", linestyle="--", label="Hot threshold (0.75)")
ax.axvline(x=0.50, color="#D4A843", linestyle="--", label="Warm threshold (0.50)")
ax.set_xlabel("Conversion Probability")
ax.set_ylabel("Active Leads")
ax.set_title("Active Lead Score Distribution")
ax.legend(fontsize=8)
ax.set_facecolor("#FDFCFA")

# 8e. Priority Distribution
ax = axes[1, 1]
priority_counts = leads_active["lead_priority"].value_counts().reindex(["Hot", "Warm", "Cool", "Cold"])
colors_priority = ["#C1666B", "#D4A843", "#5E7F9A", "#9A8C7A"]
priority_counts.plot(kind="bar", ax=ax, color=colors_priority, edgecolor="white")
ax.set_title("Active Lead Priority Distribution")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_facecolor("#FDFCFA")

# 8f. Model Comparison
ax = axes[1, 2]
model_names = list(results.keys())
aucs = [results[n]["auc"] for n in model_names]
f1s = [results[n]["f1"] for n in model_names]
x_pos = np.arange(len(model_names))
ax.bar(x_pos - 0.15, aucs, 0.3, label="ROC-AUC", color="#C9A96E")
ax.bar(x_pos + 0.15, f1s, 0.3, label="F1-Score", color="#6B8F71")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15, fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_title("Model Comparison")
ax.legend()
ax.set_facecolor("#FDFCFA")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lead_model_evaluation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: lead_model_evaluation.png")

# ============================================================================
# 9. SHAP ANALYSIS
# ============================================================================
if HAS_SHAP and HAS_XGB and "XGBoost" in results:
    print("\n--- SHAP Analysis ---")
    xgb_model = results["XGBoost"]["model"]
    explainer = shap.TreeExplainer(xgb_model)
    
    sample_size = min(1500, len(X_test))
    X_shap = X_test[:sample_size]
    shap_values = explainer.shap_values(X_shap)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#FAF8F5")
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False, max_display=15)
    plt.title("SHAP Feature Impact on Lead Conversion", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lead_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: lead_shap_summary.png")

# ============================================================================
# 10. SAVE MODELS
# ============================================================================
print("\n--- Saving Models ---")

for name, res in results.items():
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(res["model"], os.path.join(MODEL_DIR, f"lead_model_{safe_name}.joblib"))
    print(f"  Saved: lead_model_{safe_name}.joblib")

joblib.dump(scaler, os.path.join(MODEL_DIR, "lead_scaler.joblib"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "lead_label_encoders.joblib"))

with open(os.path.join(MODEL_DIR, "lead_feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)

metadata = {
    "best_model": best_model_name,
    "best_auc": round(best_auc, 4),
    "cv_auc_mean": round(cv_scores.mean(), 4),
    "cv_auc_std": round(cv_scores.std(), 4),
    "n_features": len(feature_cols),
    "training_samples": int(X_train_bal.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "total_leads": int(len(leads)),
    "active_leads_scored": int(len(leads_active)),
    "hot_leads": int((leads_active["lead_priority"] == "Hot").sum()),
    "warm_leads": int((leads_active["lead_priority"] == "Warm").sum()),
    "models": {name: {"auc": round(r["auc"], 4), "f1": round(r["f1"], 4), "accuracy": round(r["accuracy"], 4)}
               for name, r in results.items()},
}

with open(os.path.join(MODEL_DIR, "lead_model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Best Model: {best_model_name}")
print(f"ROC-AUC: {best_auc:.4f}")
print(f"5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Hot Leads: {metadata['hot_leads']} | Warm: {metadata['warm_leads']}")
print(f"All artifacts saved to: {MODEL_DIR} and {OUTPUT_DIR}")
