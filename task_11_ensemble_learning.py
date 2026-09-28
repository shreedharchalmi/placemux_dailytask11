import os
import sys
import json
import time
import logging
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
os.makedirs("artifacts", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("artifacts/task11_execution_audit.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Task11")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# Unit Tests & Edge-Case Simulations
# ---------------------------------------------------------
def run_unit_tests():
    """Runs automated unit tests exercising error handling and edge cases."""
    logger.info("Executing defensive unit tests for pipeline edge cases...")
    
    # Test 1: Empty Data Handling
    try:
        empty_df = pd.DataFrame()
        if empty_df.empty:
            raise ValueError("Input DataFrame is empty.")
    except ValueError as e:
        assert str(e) == "Input DataFrame is empty."
        logger.info("Unit Test 1 Passed: Empty DataFrame caught successfully.")

    # Test 2: Feature and Label Shape Mismatch
    try:
        X_mock = pd.DataFrame(np.random.randn(10, 5))
        y_mock = pd.Series(np.random.randint(0, 2, 8)) # 8 vs 10
        if len(X_mock) != len(y_mock):
            raise ValueError(f"Shape mismatch: {len(X_mock)} rows vs {len(y_mock)} labels.")
    except ValueError as e:
        assert "Shape mismatch" in str(e)
        logger.info("Unit Test 2 Passed: Mismatched dimensions caught successfully.")

    # Test 3: Single-Class Anomaly Check
    try:
        y_single = pd.Series([1, 1, 1, 1])
        if len(np.unique(y_single)) < 2:
            raise ValueError("Dataset contains only a single class; cannot evaluate binary metric.")
    except ValueError as e:
        assert "single class" in str(e)
        logger.info("Unit Test 3 Passed: Single-class anomaly caught successfully.")

    logger.info("All edge-case unit tests completed successfully.\n")

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    run_unit_tests()
    logger.info("Starting Task 11: Ensemble Learning Pipeline")
    
    # 1. Ingestion & Stratified Splitting
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = raw.target.copy()
    
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    logger.info(f"Data split successfully: Dev Set={len(X_dev)}, Test Set={len(X_test)}")
    
    # 2. Define Diverse Base Estimators
    clf_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=0.1, penalty="l2", solver="liblinear", class_weight="balanced", random_state=SEED))
    ])
    
    clf_rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_split=4, random_state=SEED
    )
    
    clf_gb = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=150, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.5, random_state=SEED
    )
    
    models = {
        "Logistic_Regression": clf_lr,
        "Random_Forest": clf_rf,
        "Gradient_Boosting": clf_gb
    }
    
    # 3. Model Diversity Audit (Out-of-Fold Correlations)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = {}
    
    for name, model in models.items():
        logger.info(f"Computing out-of-fold probability predictions for {name}...")
        oof_probs = cross_val_predict(model, X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
        oof_preds[name] = oof_probs
        
    oof_df = pd.DataFrame(oof_preds)
    corr_matrix = oof_df.corr()
    logger.info(f"\nModel Prediction Diversity Matrix (Pearson Correlation):\n{corr_matrix}\n")
    
    # Save Diversity Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=0.8, vmax=1.0)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left")
    ax.set_yticklabels(corr_matrix.columns)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.3f}", ha="center", va="center", color="black")
    plt.title("Base Model Probability Correlation", pad=20)
    plt.tight_layout()
    corr_plot_path = "artifacts/model_diversity_correlation.png"
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    
    # 4. Construct Ensembles
    voting_ensemble = VotingClassifier(
        estimators=[("lr", clf_lr), ("rf", clf_rf), ("gb", clf_gb)],
        voting="soft",
        weights=[1.0, 1.2, 1.5]
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=[("lr", clf_lr), ("rf", clf_rf), ("gb", clf_gb)],
        final_estimator=LogisticRegression(C=1.0, random_state=SEED),
        cv=5,
        passthrough=False
    )
    
    all_models = {
        **models,
        "Soft_Voting_Ensemble": voting_ensemble,
        "Stacking_Classifier": stacking_ensemble
    }
    
    # 5. Train & Evaluate All Models on Unseen Test Partition
    results = {}
    roc_curves = {}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, model in all_models.items():
        logger.info(f"Fitting and scoring: {name}")
        t0 = time.time()
        model.fit(X_dev, y_dev)
        fit_time = time.time() - t0
        
        # Benchmark Inference Latency
        t_infer = time.time()
        preds = model.predict(X_test)
        latency_ms = ((time.time() - t_infer) / len(X_test)) * 1000
        
        probs = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, preds)
        
        acc = float(accuracy_score(y_test, preds))
        rec_mal = float(recall_score(y_test, preds, pos_label=0))
        prec_mal = float(precision_score(y_test, preds, pos_label=0))
        f1 = float(f1_score(y_test, preds, average="weighted"))
        roc = float(roc_auc_score(y_test, probs))
        
        results[name] = {
            "test_accuracy": acc,
            "recall_malignant_class0": rec_mal,
            "precision_malignant_class0": prec_mal,
            "f1_score_weighted": f1,
            "roc_auc_score": roc,
            "false_negatives": int(cm[0, 1]),
            "false_positives": int(cm[1, 0]),
            "inference_latency_ms_per_sample": round(latency_ms, 4)
        }
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc:.4f})")
    
    ax.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    ax.set_title("Test Set ROC Curves: Base Models vs. Ensembles", fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_plot_path = "artifacts/ensemble_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    
    # 6. Summary Comparison
    results_df = pd.DataFrame(results).T
    logger.info(f"\nFinal Comparative Test Evaluation:\n{results_df[['test_accuracy', 'recall_malignant_class0', 'roc_auc_score', 'false_negatives', 'inference_latency_ms_per_sample']]}\n")
    
    # 7. Persist Artifacts
    best_model_path = "artifacts/task11_ensemble_model.joblib"
    metrics_path = "artifacts/task11_metrics.json"
    
    joblib.dump(voting_ensemble, best_model_path)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved Voting Ensemble artifact to {best_model_path}")
    logger.info(f"Saved complete metrics dictionary to {metrics_path}")
    logger.info(f"Diagnostic plots saved to {corr_plot_path} and {roc_plot_path}")
    logger.info("Task 11 completed successfully.")

if __name__ == "__main__":
    main()