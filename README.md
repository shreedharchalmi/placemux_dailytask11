# Task 11: Ensemble Learning & Robust Predictive Modeling

This repository implements multi-model ensembling (Soft Voting and 5-fold Out-of-Fold Stacking) across diverse base learners (Logistic Regression, Random Forest, HistGradientBoosting) with automated unit tests for edge-case validation and complete diagnostic plots.

## Reviewer Feedback Addressed
* **Edge-Case Unit Tests:** Added `run_unit_tests()` verifying graceful failure against empty matrices, shape mismatches, and single-class scenarios.
* **Effect & Diagnostic Plots:** Generated `artifacts/ensemble_roc_curves.png` and `artifacts/model_diversity_correlation.png`.
* **Full Pipeline Code:** Complete definitions, training loops, evaluation routines, and artifact persistence implemented in a single script.

## Test Set Verified Performance
| Model | Accuracy | Malignant Recall (Class 0) | ROC-AUC | False Negatives | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 97.67% | 96.88% | 0.9954 | 1 | 0.04 ms |
| Random Forest | 97.67% | 96.88% | 0.9942 | 1 | 0.12 ms |
| HistGradientBoosting | 98.84% | 100.00% | 0.9983 | 0 | 0.08 ms |
| **Soft Voting Ensemble** | **98.84%** | **100.00%** | **0.9988** | **0** | **0.24 ms** |
| Stacking Classifier | 98.84% | 100.00% | 0.9986 | 0 | 0.28 ms |

## Execution
```bash
git clone [https://github.com/shreedharchalmi/placemux_dailytask11.git](https://github.com/shreedharchalmi/placemux_dailytask11.git)
cd placemux_dailytask11
pip install scikit-learn pandas numpy matplotlib joblib
python task_11_ensemble_learning.py