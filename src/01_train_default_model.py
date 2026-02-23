#!/usr/bin/env python3
"""
============================================================================
ROSHN Community Intelligence - Payment Default Prediction Model
============================================================================
Enterprise AI for Real Estate Operations
ML Intelligence & Predictive Analytics Module

Author  : Sreekrishnan
Version : 1.0
Purpose : Train, evaluate, and export payment default prediction models
          for the ROSHN Community Intelligence Command Center
============================================================================

HOW TO RUN:
    1. Place your CSV files in the /data folder:
       - roshn_residents_master.csv
       - roshn_payment_transactions.csv
       - roshn_complaints.csv
       - roshn_ai_interactions.csv
    
    2. Install dependencies:
       pip install -r requirements.txt
    
    3. Run this script:
       python src/01_train_default_model.py
    
    4. Outputs will be saved to /models and /outputs
============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
import joblib
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    average_precision_score, log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not installed. Run: pip install shap")

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("⚠️  imbalanced-learn not installed. Run: pip install imbalanced-learn")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "default_flag"

# ROSHN Brand Colors
ROSHN_COLORS = {
    "primary": "#1B3A4B",       # Deep navy
    "secondary": "#4ECDC4",     # Teal
    "accent": "#FF6B6B",        # Coral red
    "warning": "#FFE66D",       # Yellow
    "success": "#2ECC71",       # Green
    "danger": "#E74C3C",        # Red
    "bg_dark": "#0D1B2A",      # Dark background
    "bg_light": "#F7F9FC",     # Light background
    "text": "#2C3E50",         # Dark text
    "grid": "#E8E8E8",         # Grid lines
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
def load_data():
    """Load all datasets and perform initial validation."""
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    required_files = {
        "residents": "roshn_residents_master.csv",
        "payments": "roshn_payment_transactions.csv",
        "complaints": "roshn_complaints.csv",
        "interactions": "roshn_ai_interactions.csv",
    }
    
    datasets = {}
    for key, filename in required_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"   ❌ Missing: {filename}")
            print(f"      Please place the file in: {DATA_DIR}")
            sys.exit(1)
        
        df = pd.read_csv(filepath)
        datasets[key] = df
        print(f"   ✅ {filename:<40} : {len(df):>10,} rows × {len(df.columns)} cols")
    
    return datasets


# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
def engineer_features(datasets):
    """Create ML-ready features from raw data."""
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    df = datasets["residents"].copy()
    df_payments = datasets["payments"].copy()
    df_complaints = datasets["complaints"].copy()
    df_interactions = datasets["interactions"].copy()
    
    # ---- 2a. Payment Behavior Features ----
    print("   [2a] Engineering payment behavior features...")
    
    pay_agg = df_payments.groupby("resident_id").agg(
        total_payments=("payment_id", "count"),
        avg_amount_due=("amount_due_aed", "mean"),
        avg_delay_days=("delay_days", "mean"),
        max_delay_days=("delay_days", "max"),
    ).reset_index()
    
    # Add heavy noise to payment features to reduce leakage
    noise_scale = 1.2  # Heavy noise injection
    pay_agg["avg_delay_days"] = (
        pay_agg["avg_delay_days"].fillna(0) + 
        np.random.normal(0, pay_agg["avg_delay_days"].fillna(0).std() * noise_scale, len(pay_agg))
    ).clip(0)
    pay_agg["max_delay_days"] = (
        pay_agg["max_delay_days"].fillna(0) + 
        np.random.normal(0, pay_agg["max_delay_days"].fillna(0).std() * noise_scale, len(pay_agg))
    ).clip(0)
    
    # Late payment count (with noise)
    paid_late = df_payments[df_payments["payment_status"] == "Paid Late"].groupby("resident_id").size().reset_index(name="pay_late_count")
    
    # Merge
    df = df.merge(pay_agg, on="resident_id", how="left")
    df = df.merge(paid_late, on="resident_id", how="left")
    df["pay_late_count"] = df["pay_late_count"].fillna(0)
    
    # ---- 2b. Complaint Features ----
    print("   [2b] Engineering complaint features...")
    
    comp_agg = df_complaints.groupby("resident_id").agg(
        complaint_count=("complaint_id", "count"),
        avg_resolution_hours=("resolution_hours", "mean"),
        max_resolution_hours=("resolution_hours", "max"),
        avg_satisfaction_rating=("satisfaction_rating", "mean"),
    ).reset_index()
    
    # Severity distribution
    comp_severity = df_complaints.groupby(["resident_id", "severity"]).size().unstack(fill_value=0).reset_index()
    comp_severity.columns = ["resident_id"] + [f"complaint_{c.lower()}" for c in comp_severity.columns[1:]]
    
    # Complaint status
    comp_status = df_complaints.groupby(["resident_id", "status"]).size().unstack(fill_value=0).reset_index()
    comp_status.columns = ["resident_id"] + [f"comp_status_{c.lower().replace(' ', '_')}" for c in comp_status.columns[1:]]
    
    df = df.merge(comp_agg, on="resident_id", how="left")
    df = df.merge(comp_severity, on="resident_id", how="left")
    df = df.merge(comp_status, on="resident_id", how="left")
    
    # ---- 2c. AI Interaction Features ----
    print("   [2c] Engineering AI interaction features...")
    
    int_agg = df_interactions.groupby("resident_id").agg(
        interaction_count=("interaction_id", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        min_sentiment=("sentiment_score", "min"),
        max_sentiment=("sentiment_score", "max"),
        avg_duration_sec=("duration_seconds", "mean"),
        ai_resolved_count=("resolved_by_ai", "sum"),
        escalated_count=("escalated_to_human", "sum"),
        avg_csat=("csat_score", "mean"),
    ).reset_index()
    
    int_agg["ai_resolution_rate"] = (int_agg["ai_resolved_count"] / int_agg["interaction_count"].replace(0, 1))
    int_agg["escalation_rate"] = (int_agg["escalated_count"] / int_agg["interaction_count"].replace(0, 1))
    
    # Channel distribution
    int_channel = df_interactions.groupby(["resident_id", "channel"]).size().unstack(fill_value=0).reset_index()
    int_channel.columns = ["resident_id"] + [f"channel_{c.lower().replace(' ', '_').replace('/', '_')}" for c in int_channel.columns[1:]]
    
    df = df.merge(int_agg, on="resident_id", how="left")
    df = df.merge(int_channel, on="resident_id", how="left")
    
    # ---- 2d. Derived Features ----
    print("   [2d] Computing derived features...")
    
    # Payment stress index (using only non-leaky features)
    df["payment_stress_index"] = (
        df["debt_to_income_pct"] * 0.5 +
        (900 - df["credit_score"]) / 600 * 30 +
        df.get("complaint_count", pd.Series(0, index=df.index)).fillna(0).clip(upper=10) / 10 * 20
    ).round(2)
    
    # Engagement score
    df["engagement_score"] = (
        df.get("interaction_count", pd.Series(0, index=df.index)).fillna(0) * 2 +
        df.get("complaint_count", pd.Series(0, index=df.index)).fillna(0) * (-1) +
        df["tenure_months"] * 0.1
    ).round(2)
    
    # Income adequacy ratio
    df["income_adequacy_ratio"] = np.where(
        df["monthly_installment_aed"] > 0,
        df["monthly_income_aed"] / df["monthly_installment_aed"],
        10
    )
    df["income_adequacy_ratio"] = df["income_adequacy_ratio"].clip(0, 50).round(2)
    
    # Credit utilization proxy
    df["balance_to_value_ratio"] = (
        df["outstanding_balance_aed"] / df["property_value_aed"].replace(0, 1)
    ).round(4)
    
    # Fill NAs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"\n   📊 Final feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


# ============================================================================
# STEP 3: PREPARE TRAINING DATA
# ============================================================================
def prepare_training_data(df):
    """Select features and prepare train/test split."""
    print("\n" + "=" * 70)
    print("STEP 3: PREPARING TRAINING DATA")
    print("=" * 70)
    
    # Drop non-feature columns and leaky features
    drop_cols = [
        "resident_id", "unit_id", "first_name", "last_name", "phone", "email",
        "move_in_date", "risk_score", "risk_category", "satisfaction_score",
        "total_complaints", "avg_sentiment", "avg_csat",  # Pre-computed aggregates
        "current_dpd", "max_dpd_12m",  # Directly generated from default flag
        "payment_consistency_pct",      # Directly generated from default flag
        "late_payments_12m",            # Directly generated from default flag
        TARGET_COL
    ]
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Encode categorical variables
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    le_dict = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    
    print(f"   Features selected        : {len(feature_cols)}")
    print(f"   Categorical encoded      : {len(cat_cols)} columns")
    print(f"   Target distribution      :")
    print(f"     Non-Default (0)        : {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
    print(f"     Default    (1)         : {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n   Train set                : {len(X_train):,} samples")
    print(f"   Test set                 : {len(X_test):,} samples")
    
    # Apply SMOTE for class balancing
    if HAS_IMBLEARN:
        print("   Applying SMOTE oversampling...")
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.4)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        print(f"   After SMOTE              : {len(X_train_bal):,} samples")
        print(f"     Non-Default            : {(y_train_bal == 0).sum():,}")
        print(f"     Default                : {(y_train_bal == 1).sum():,}")
    else:
        X_train_bal, y_train_bal = X_train, y_train
        print("   ⚠️  SMOTE not available, using original class distribution")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Save artifacts
    feature_info = {
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols,
        "num_features": len(feature_cols),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    
    return (X_train_scaled, X_test_scaled, y_train_bal, y_test,
            X_train_bal, X_test,  # Unscaled (SMOTE-resampled) for tree models
            scaler, le_dict, feature_info, feature_cols)


# ============================================================================
# STEP 4: TRAIN MODELS
# ============================================================================
def train_models(X_train_scaled, X_test_scaled, y_train, y_test,
                 X_train_raw, X_test_raw, feature_cols):
    """Train multiple models and compare performance."""
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING MODELS")
    print("=" * 70)
    
    results = {}
    trained_models = {}
    
    # ---- Model 1: Logistic Regression ----
    print("\n   📌 Model 1: Logistic Regression")
    print("   " + "-" * 50)
    
    lr = LogisticRegression(
        max_iter=1000,
        C=0.5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    lr.fit(X_train_scaled, y_train)
    
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_ap = average_precision_score(y_test, lr_prob)
    
    results["Logistic Regression"] = {
        "ROC-AUC": lr_auc, "F1": lr_f1, "Accuracy": lr_acc, "Avg Precision": lr_ap
    }
    trained_models["logistic_regression"] = lr
    
    print(f"   ROC-AUC  : {lr_auc:.4f}")
    print(f"   F1-Score : {lr_f1:.4f}")
    print(f"   Accuracy : {lr_acc:.4f}")
    
    # ---- Model 2: Random Forest ----
    print("\n   📌 Model 2: Random Forest")
    print("   " + "-" * 50)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_raw, y_train)
    
    rf_pred = rf.predict(X_test_raw)
    rf_prob = rf.predict_proba(X_test_raw)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_ap = average_precision_score(y_test, rf_prob)
    
    results["Random Forest"] = {
        "ROC-AUC": rf_auc, "F1": rf_f1, "Accuracy": rf_acc, "Avg Precision": rf_ap
    }
    trained_models["random_forest"] = rf
    
    print(f"   ROC-AUC  : {rf_auc:.4f}")
    print(f"   F1-Score : {rf_f1:.4f}")
    print(f"   Accuracy : {rf_acc:.4f}")
    
    # ---- Model 3: Gradient Boosting ----
    print("\n   📌 Model 3: Gradient Boosting")
    print("   " + "-" * 50)
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train_raw, y_train)
    
    gb_pred = gb.predict(X_test_raw)
    gb_prob = gb.predict_proba(X_test_raw)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_prob)
    gb_f1 = f1_score(y_test, gb_pred)
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_ap = average_precision_score(y_test, gb_prob)
    
    results["Gradient Boosting"] = {
        "ROC-AUC": gb_auc, "F1": gb_f1, "Accuracy": gb_acc, "Avg Precision": gb_ap
    }
    trained_models["gradient_boosting"] = gb
    
    print(f"   ROC-AUC  : {gb_auc:.4f}")
    print(f"   F1-Score : {gb_f1:.4f}")
    print(f"   Accuracy : {gb_acc:.4f}")
    
    # ---- Model 4: XGBoost ----
    if HAS_XGBOOST:
        print("\n   📌 Model 4: XGBoost")
        print("   " + "-" * 50)
        
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            use_label_encoder=False,
            verbosity=0
        )
        xgb_model.fit(X_train_raw, y_train)
        
        xgb_pred = xgb_model.predict(X_test_raw)
        xgb_prob = xgb_model.predict_proba(X_test_raw)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_prob)
        xgb_f1 = f1_score(y_test, xgb_pred)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_ap = average_precision_score(y_test, xgb_prob)
        
        results["XGBoost"] = {
            "ROC-AUC": xgb_auc, "F1": xgb_f1, "Accuracy": xgb_acc, "Avg Precision": xgb_ap
        }
        trained_models["xgboost"] = xgb_model
        
        print(f"   ROC-AUC  : {xgb_auc:.4f}")
        print(f"   F1-Score : {xgb_f1:.4f}")
        print(f"   Accuracy : {xgb_acc:.4f}")
    
    # ---- Cross-Validation for best model ----
    print("\n   📌 Cross-Validation (5-Fold)")
    print("   " + "-" * 50)
    
    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    best_model_key = best_model_name.lower().replace(" ", "_")
    best_model = trained_models[best_model_key]
    
    # Use appropriate data for CV
    if best_model_key in ["logistic_regression"]:
        cv_X = X_train_scaled
    else:
        cv_X = X_train_raw
    
    cv_scores = cross_val_score(best_model, cv_X, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"   Best Model: {best_model_name}")
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   CV Scores : {[f'{s:.4f}' for s in cv_scores]}")
    
    # ---- Model Comparison Summary ----
    print("\n   " + "=" * 55)
    print(f"   {'MODEL COMPARISON SUMMARY':^55}")
    print("   " + "=" * 55)
    print(f"   {'Model':<25} {'ROC-AUC':>10} {'F1':>10} {'Accuracy':>10}")
    print("   " + "-" * 55)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["ROC-AUC"], reverse=True):
        marker = " ⭐" if name == best_model_name else ""
        print(f"   {name:<25} {metrics['ROC-AUC']:>10.4f} {metrics['F1']:>10.4f} {metrics['Accuracy']:>10.4f}{marker}")
    print("   " + "=" * 55)
    
    return results, trained_models, best_model_name


# ============================================================================
# STEP 5: GENERATE EVALUATION PLOTS
# ============================================================================
def generate_plots(trained_models, results, X_test_scaled, X_test_raw, y_test, feature_cols):
    """Generate comprehensive evaluation charts."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING EVALUATION PLOTS")
    print("=" * 70)
    
    # ---- 5a. ROC Curves ----
    print("   [5a] ROC Curves...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = [ROSHN_COLORS["primary"], ROSHN_COLORS["secondary"], 
              ROSHN_COLORS["accent"], "#9B59B6"]
    
    for idx, (key, model) in enumerate(trained_models.items()):
        X_eval = X_test_scaled if key == "logistic_regression" else X_test_raw
        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        name = key.replace("_", " ").title()
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.5,
                label=f"{name} (AUC = {auc:.4f})")
    
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Payment Default Prediction Models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- 5b. Feature Importance (Best Tree Model) ----
    print("   [5b] Feature Importance...")
    
    tree_model_key = None
    for key in ["xgboost", "gradient_boosting", "random_forest"]:
        if key in trained_models:
            tree_model_key = key
            break
    
    if tree_model_key:
        model = trained_models[tree_model_key]
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=True).tail(20)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"], 
                       color=ROSHN_COLORS["secondary"], edgecolor=ROSHN_COLORS["primary"], lw=0.5)
        
        # Highlight top 5
        for bar in bars[-5:]:
            bar.set_color(ROSHN_COLORS["primary"])
        
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top 20 Features - {tree_model_key.replace('_', ' ').title()}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "02_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        # Save feature importance to CSV
        feat_imp_full = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        feat_imp_full.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    # ---- 5c. Confusion Matrix (Best Model) ----
    print("   [5c] Confusion Matrix...")
    
    best_key = max(results, key=lambda k: results[k]["ROC-AUC"]).lower().replace(" ", "_")
    best_model = trained_models[best_key]
    X_eval = X_test_scaled if best_key == "logistic_regression" else X_test_raw
    y_pred = best_model.predict(X_eval)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", 
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"],
                annot_kws={"size": 16}, ax=ax)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"Confusion Matrix - {best_key.replace('_', ' ').title()}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- 5d. Precision-Recall Curve ----
    print("   [5d] Precision-Recall Curve...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for idx, (key, model) in enumerate(trained_models.items()):
        X_eval = X_test_scaled if key == "logistic_regression" else X_test_raw
        y_prob = model.predict_proba(X_eval)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        name = key.replace("_", " ").title()
        ax.plot(recall, precision, color=colors[idx % len(colors)], lw=2.5,
                label=f"{name} (AP = {ap:.4f})")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_precision_recall.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- 5e. Risk Score Distribution ----
    print("   [5e] Risk Score Distribution...")
    
    y_prob = best_model.predict_proba(X_eval)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution by class
    axes[0].hist(y_prob[y_test == 0], bins=50, alpha=0.7, color=ROSHN_COLORS["success"], 
                 label="Non-Default", density=True)
    axes[0].hist(y_prob[y_test == 1], bins=50, alpha=0.7, color=ROSHN_COLORS["danger"], 
                 label="Default", density=True)
    axes[0].set_xlabel("Predicted Default Probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution by Class", fontweight="bold")
    axes[0].legend()
    
    # Overall distribution with risk zones
    axes[1].hist(y_prob, bins=50, color=ROSHN_COLORS["primary"], alpha=0.8, density=True)
    axes[1].axvline(0.2, color=ROSHN_COLORS["success"], ls="--", lw=2, label="Low Risk (<0.2)")
    axes[1].axvline(0.5, color=ROSHN_COLORS["warning"], ls="--", lw=2, label="Medium Risk (0.2-0.5)")
    axes[1].axvline(0.8, color=ROSHN_COLORS["danger"], ls="--", lw=2, label="High Risk (>0.5)")
    axes[1].set_xlabel("Predicted Default Probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Risk Score Distribution", fontweight="bold")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_risk_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- 5f. Model Comparison Bar Chart ----
    print("   [5f] Model Comparison Chart...")
    
    df_results = pd.DataFrame(results).T
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(df_results))
    width = 0.22
    
    ax.bar(x - width*1.5, df_results["ROC-AUC"], width, label="ROC-AUC", 
           color=ROSHN_COLORS["primary"], edgecolor="white")
    ax.bar(x - width*0.5, df_results["F1"], width, label="F1-Score", 
           color=ROSHN_COLORS["secondary"], edgecolor="white")
    ax.bar(x + width*0.5, df_results["Accuracy"], width, label="Accuracy", 
           color=ROSHN_COLORS["accent"], edgecolor="white")
    ax.bar(x + width*1.5, df_results["Avg Precision"], width, label="Avg Precision", 
           color="#9B59B6", edgecolor="white")
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_results.index, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print("   ✅ All plots saved to /outputs/")
    
    return best_key


# ============================================================================
# STEP 6: SHAP EXPLAINABILITY
# ============================================================================
def generate_shap_analysis(trained_models, X_test_raw, feature_cols, best_key):
    """Generate SHAP explanations for model interpretability."""
    print("\n" + "=" * 70)
    print("STEP 6: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    
    if not HAS_SHAP:
        print("   ⚠️  SHAP not installed. Skipping explainability analysis.")
        print("   Install with: pip install shap")
        return None
    
    # Use tree model for SHAP (much faster)
    tree_key = None
    for key in ["xgboost", "gradient_boosting", "random_forest"]:
        if key in trained_models:
            tree_key = key
            break
    
    if tree_key is None:
        print("   ⚠️  No tree model available for SHAP.")
        return None
    
    model = trained_models[tree_key]
    print(f"   Using {tree_key.replace('_', ' ').title()} for SHAP analysis...")
    
    # Sample for speed
    sample_size = min(2000, len(X_test_raw))
    X_sample = X_test_raw.iloc[:sample_size]
    
    print(f"   Computing SHAP values for {sample_size} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Class 1 (default)
    else:
        shap_vals = shap_values
    
    # ---- SHAP Summary Plot ----
    print("   [6a] SHAP Summary Plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_cols, 
                      show=False, max_display=20)
    plt.title("SHAP Feature Impact on Default Prediction", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- SHAP Bar Plot ----
    print("   [6b] SHAP Importance Bar Plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_cols, 
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Mean Absolute Impact", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # ---- Save SHAP values ----
    shap_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": np.abs(shap_vals).mean(axis=0)
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    shap_importance.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
    
    print("   ✅ SHAP analysis complete")
    
    return shap_vals


# ============================================================================
# STEP 7: SAVE MODELS & ARTIFACTS
# ============================================================================
def save_artifacts(trained_models, scaler, le_dict, feature_info, results, 
                   feature_cols, best_model_name, df):
    """Save all models, scalers, and metadata for dashboard use."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVING MODELS & ARTIFACTS")
    print("=" * 70)
    
    # Save all trained models
    for key, model in trained_models.items():
        path = os.path.join(MODEL_DIR, f"model_{key}.joblib")
        joblib.dump(model, path)
        print(f"   ✅ Saved: model_{key}.joblib")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"   ✅ Saved: scaler.joblib")
    
    # Save label encoders
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    print(f"   ✅ Saved: label_encoders.joblib")
    
    # Save feature list
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"   ✅ Saved: feature_columns.json")
    
    # Save model metadata
    best_key = best_model_name.lower().replace(" ", "_")
    metadata = {
        "project": "ROSHN Community Intelligence - Payment Default Prediction",
        "version": "1.0",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": feature_info["train_size"] + feature_info["test_size"],
        "num_features": feature_info["num_features"],
        "best_model": best_model_name,
        "model_results": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in results.items()},
        "best_roc_auc": round(results[best_model_name]["ROC-AUC"], 4),
        "best_f1": round(results[best_model_name]["F1"], 4),
        "default_rate": round(df[TARGET_COL].mean(), 4),
        "risk_distribution": df["risk_category"].value_counts().to_dict(),
    }
    
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"   ✅ Saved: model_metadata.json")
    
    # Save enriched residents data with predictions
    best_model = trained_models[best_key]
    
    # We need the full feature matrix to score all residents
    print(f"\n   Scoring all {len(df)} residents with {best_model_name}...")
    
    drop_cols = [
        "resident_id", "unit_id", "first_name", "last_name", "phone", "email",
        "move_in_date", "risk_score", "risk_category", "satisfaction_score",
        "total_complaints", "avg_sentiment", "avg_csat", TARGET_COL
    ]
    X_all = df[[c for c in feature_cols if c in df.columns]].copy()
    
    # Fill any missing columns
    for col in feature_cols:
        if col not in X_all.columns:
            X_all[col] = 0
    X_all = X_all[feature_cols]
    
    if best_key == "logistic_regression":
        X_all_eval = pd.DataFrame(scaler.transform(X_all), columns=feature_cols)
    else:
        X_all_eval = X_all
    
    df["predicted_default_prob"] = best_model.predict_proba(X_all_eval)[:, 1]
    df["predicted_default_prob"] = df["predicted_default_prob"].round(4)
    df["predicted_risk_grade"] = pd.cut(
        df["predicted_default_prob"],
        bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
        labels=["A - Very Low", "B - Low", "C - Medium", "D - High", "E - Critical"]
    )
    
    # Save scored dataset
    scored_path = os.path.join(OUTPUT_DIR, "roshn_residents_scored.csv")
    df.to_csv(scored_path, index=False)
    print(f"   ✅ Saved: roshn_residents_scored.csv ({len(df)} residents with predictions)")
    
    # Print risk grade distribution
    print(f"\n   📊 PREDICTED RISK GRADE DISTRIBUTION:")
    for grade, count in df["predicted_risk_grade"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"      {grade:<20} : {count:>6,} residents ({pct:.1f}%)")
    
    return metadata


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ROSHN COMMUNITY INTELLIGENCE".center(68) + "█")
    print("█" + "  Payment Default Prediction Model".center(68) + "█")
    print("█" + "  ML Intelligence & Predictive Analytics".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print(f"\n  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project   : {PROJECT_DIR}")
    print(f"  Data Dir  : {DATA_DIR}")
    print(f"  Model Dir : {MODEL_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    
    # Run pipeline
    datasets = load_data()
    df = engineer_features(datasets)
    
    (X_train_scaled, X_test_scaled, y_train, y_test,
     X_train_raw, X_test_raw,
     scaler, le_dict, feature_info, feature_cols) = prepare_training_data(df)
    
    results, trained_models, best_model_name = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test,
        X_train_raw, X_test_raw, feature_cols
    )
    
    best_key = generate_plots(
        trained_models, results, X_test_scaled, X_test_raw, y_test, feature_cols
    )
    
    shap_vals = generate_shap_analysis(trained_models, X_test_raw, feature_cols, best_key)
    
    metadata = save_artifacts(
        trained_models, scaler, le_dict, feature_info, results,
        feature_cols, best_model_name, df
    )
    
    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"""
   🏆 Best Model          : {best_model_name}
   📈 ROC-AUC             : {metadata['best_roc_auc']:.4f}
   🎯 F1-Score            : {metadata['best_f1']:.4f}
   📊 Features Used       : {metadata['num_features']}
   👥 Dataset Size        : {metadata['dataset_size']:,}
   💰 Default Rate        : {metadata['default_rate']*100:.1f}%
   
   📁 Models saved to     : {MODEL_DIR}/
   📁 Outputs saved to    : {OUTPUT_DIR}/
   
   NEXT STEP: Build the Streamlit Dashboard
   Run: python src/02_dashboard.py (coming next)
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
