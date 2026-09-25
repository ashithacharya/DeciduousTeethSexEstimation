"""
Stepwise Forward Feature Selection and Model Evaluation
Implemented on 14 Aug 2026
Author: Ashith Acharya
Description:
    Performs 5-fold cross-validation-based stepwise forward feature selection
    on Tooth_Dec_Dimensions_400.csv, identifies optimal subsets of tooth dimensions,
    and evaluates Gradient Boosting, CatBoost, Extra Trees, LightGBM, and Logistic Regression
    on Tooth_Dec_Dimensions_100.csv.
"""
# 1. Install required packages in Colab environment
!pip install catboost lightgbm xgboost -q

import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 2. Universal Seed Setup for Strict Reproducibility
SEED = 42


def seed_everything(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)


seed_everything(SEED)

# 3. Import Machine Learning Modules
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

# 4. Load Datasets & Clean Artifacts/NaNs
train_path = "Tooth_Dec_Dimensions_400.csv"
test_path = "Tooth_Dec_Dimensions_100.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Automatically drop ghost index columns (e.g., 'Unnamed: 41')
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

# Remove completely empty rows or columns if present
train_df = train_df.dropna(how="all", axis=1).dropna(how="all", axis=0)
test_df = test_df.dropna(how="all", axis=1).dropna(how="all", axis=0)

# Automatically identify target column ('Sex' or last column)
target_col = "Sex" if "Sex" in train_df.columns else train_df.columns[-1]

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Assert data cleanliness before running feature selection
assert not X_train.isnull().values.any(), (
    "Error: NaNs detected in training features!"
)
assert not X_test.isnull().values.any(), "Error: NaNs detected in test features!"

print("Dataset Loaded & Cleaned Successfully:")
print(f"Cleaned Training shape: X={X_train.shape}, y={y_train.shape}")
print(f"Cleaned Test shape:     X={X_test.shape}, y={y_test.shape}\n")

# 5. Define Deterministic Stratified CV Splitter
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 6. Initialize Base Models for Selection & Tuning
base_models = {
    "Logistic Regression": LogisticRegression(
        random_state=SEED, max_iter=1000
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "CatBoost": CatBoostClassifier(random_state=SEED, verbose=0),
    "Extra Trees": ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1),
}

# 7. Define Hyperparameter Tuning Grids
param_grids = {
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
        "solver": ["liblinear", "lbfgs"],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 7],
        "min_samples_split": [2, 5, 10],
    },
    "CatBoost": {
        "iterations": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7],
    },
    "Extra Trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, -1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 30],
    },
}


# 8. Model-Specific Stepwise Forward Feature Selection Function
def model_specific_stepwise_selection(model, X, y, cv):
  best_features = []
  remaining_features = list(X.columns)
  best_score = 0.0

  while remaining_features:
    scores_with_candidates = []

    for candidate in remaining_features:
      current_features = best_features + [candidate]
      X_subset = X[current_features]

      # Evaluate candidate set using 5-Fold Stratified Cross-Validation
      cv_scores = cross_val_score(
          model, X_subset, y, cv=cv, scoring="f1_macro", n_jobs=-1
      )
      mean_score = np.mean(cv_scores)
      scores_with_candidates.append((mean_score, candidate))

    scores_with_candidates.sort(reverse=True)
    best_candidate_score, best_candidate = scores_with_candidates[0]

    if best_candidate_score > best_score:
      best_score = best_candidate_score
      best_features.append(best_candidate)
      remaining_features.remove(best_candidate)
    else:
      break

  return best_features, best_score


# 9. Execute Pipeline
test_results = []
model_selected_features = {}

print(
    "=============================================================================="
)
print(
    "      MODEL-SPECIFIC FEATURE SELECTION, TUNING & EVALUATION                   "
)
print(
    "=============================================================================="
)

for name, model in base_models.items():
  print(f"\n--- Processing Model: {name} ---")

  # Phase A: Model-Specific Feature Selection
  selected_teeth, best_cv_f1 = model_specific_stepwise_selection(
      model, X_train, y_train, cv=cv_splitter
  )
  model_selected_features[name] = selected_teeth

  print(
      f"✔ Selected {len(selected_teeth)} Tooth Dimensions (CV F1-Macro:"
      f" {best_cv_f1:.4f}):"
  )
  print(f"  Teeth: {selected_teeth}")

  # Subset features specifically for this model
  X_tr_sub = X_train[selected_teeth]
  X_te_sub = X_test[selected_teeth]

  # Phase B: Hyperparameter Tuning on Model's Custom Subset
  rs = RandomizedSearchCV(
      estimator=model,
      param_distributions=param_grids[name],
      n_iter=30,
      scoring="f1_macro",
      cv=cv_splitter,
      random_state=SEED,
      n_jobs=-1,
  )
  rs.fit(X_tr_sub, y_train)

  best_estimator = rs.best_estimator_
  best_estimator.fit(X_tr_sub, y_train)

  # Phase C: Evaluate on Held-Out Test Set (100 Dentitions)
  y_pred = best_estimator.predict(X_te_sub)
  y_proba = best_estimator.predict_proba(X_te_sub)[:, 1]

  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

  test_results.append({
      "Model": name,
      "Num_Teeth": len(selected_teeth),
      "Selected_Teeth": ", ".join(selected_teeth),
      "Accuracy": accuracy_score(y_test, y_pred),
      "Precision (Macro)": precision_score(y_test, y_pred, average="macro"),
      "Recall (Macro)": recall_score(y_test, y_pred, average="macro"),
      "F1 Score (Macro)": f1_score(y_test, y_pred, average="macro"),
      "Sensitivity": recall_score(y_test, y_pred),
      "Specificity": tn / (tn + fp),
      "ROC AUC": roc_auc_score(y_test, y_proba),
      "Average Precision": average_precision_score(y_test, y_proba),
      "Log Loss": log_loss(y_test, y_proba),
      "Brier Score": brier_score_loss(y_test, y_proba),
  })

# 10. Summary Performance Table Output
results_df = pd.DataFrame(test_results).sort_values(
    by="Accuracy", ascending=False
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

print(
    "\n=========================================================================================================="
)
print(
    "               HELD-OUT TEST SET EVALUATION (MODEL-SPECIFIC FEATURE"
    " SUBSETS)                             "
)
print(
    "=========================================================================================================="
)
print(results_df.to_string(index=False))
