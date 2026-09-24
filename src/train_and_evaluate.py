# 1. Install required packages (if needed in Colab environment)

import os
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 2. Universal Seed Setup for Strict Reproducibility
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

# 3. Import Machine Learning Modules
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 4. Load Datasets
train_path = 'Tooth_Dec_Dimensions_400.csv'
test_path = 'Tooth_Dec_Dimensions_100.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Explicitly set target as FIRST column
target_col = train_df.columns[0]

# Drop target column from features
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Align columns between train and test
common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

print(f"Dataset Loaded Successfully:")
print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape:     X={X_test.shape}, y={y_test.shape}\n")
# 5. Define Deterministic Stratified CV
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# 6. Initialize Seeded Base Models
base_models = {
    "Logistic Regression": LogisticRegression(random_state=SEED, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Support Vector Machine": SVC(probability=True, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(random_state=SEED),
    "XGBoost": XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
    "CatBoost": CatBoostClassifier(random_state=SEED, verbose=0)
}

# 7. Define Hyperparameter Grids for Optimization
param_grids = {
    "Logistic Regression": {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    },
    "Decision Tree": {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    "Support Vector Machine": {
        'C': [0.1, 1.0, 5.0, 10.0, 20.0],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    "Random Forest": {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, -1],
        'num_leaves': [15, 31, 63, 100],
        'min_child_samples': [10, 20, 30]
    },
    "Extra Trees": {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "CatBoost": {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 7]
    }
}

# 8. Execute Two-Phase Optimization and Train
best_models = {}
print("--- Executing Seeded Hyperparameter Tuning ---")

for name, model in base_models.items():
    # Phase 1: Randomized Search
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[name],
        n_iter=50,
        scoring='f1_macro',
        cv=cv_splitter,
        random_state=SEED,
        n_jobs=-1
    )
    rs.fit(X_train, y_train)
    
    # Fit final tuned estimator on full training set
    best_estimator = rs.best_estimator_
    best_estimator.fit(X_train, y_train)
    best_models[name] = best_estimator
    print(f"✔ Optimized {name}")

# 9. Evaluate on Held-Out Test Set
test_results = []

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    test_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Macro)': precision_score(y_test, y_pred, average='macro'),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro'),
        'F1 Score (Macro)': f1_score(y_test, y_pred, average='macro'),
        'Sensitivity': recall_score(y_test, y_pred),
        'Specificity': tn / (tn + fp),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'Average Precision': average_precision_score(y_test, y_proba),
        'Log Loss': log_loss(y_test, y_proba),
        'Brier Score': brier_score_loss(y_test, y_proba)
    })

# 10. Output Results Table
results_df = pd.DataFrame(test_results).sort_values(by='Accuracy', ascending=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n==============================================================================")
print("                           HELD-OUT TEST SET EVALUATION                       ")
print("==============================================================================")
print(results_df.to_string(index=False))