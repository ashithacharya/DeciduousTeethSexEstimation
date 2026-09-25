# Deciduous Teeth Sex Estimation

## Purpose
This project evaluates whether advanced machine learning algorithms outperform conventional logistic regression in sex estimation using calliper‑derived measurements of the deciduous dentition.

We train and evaluate **9 supervised ML models** (Decision Tree, SVM, Gradient Boosting, Random Forest, AdaBoost, XGBoost, LightGBM, Extra Trees, CatBoost) against a logistic regression baseline.  
Performance is assessed on an **independent held‑out test set** using classification and probability calibration metrics.

---

## Dataset
- **Training set**: `Tooth_Dec_Dimensions_400.csv` (400 cases, Sex + tooth dimensions)  
- **Test set**: `Tooth_Dec_Dimensions_100.csv` (100 cases, Sex + tooth dimensions)  
- Target variable: `Sex` (0 = female, 1 = male)

⚠️ Note: These datasets are anonymized and contain only numeric measurements.

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/DeciduousTeethSexEstimation.git
cd DeciduousTeethSexEstimation
pip install -r requirements.txt

## Usage
Run the training and evaluation script:

```bash
python src/train_and_evaluate.py
---
---
---
---
## Evaluation Results
| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) | ROC AUC |
|-------|----------|-------------------|----------------|------------------|---------|
| Gradient Boosting | 0.62 | 0.63 | 0.62 | 0.61 | 0.65 |
| CatBoost | 0.60 | 0.61 | 0.60 | 0.59 | 0.66 |
| Extra Trees | 0.59 | 0.59 | 0.59 | 0.59 | 0.66 |
| Random Forest | 0.57 | 0.57 | 0.57 | 0.57 | 0.63 |
| LightGBM | 0.56 | 0.56 | 0.56 | 0.56 | 0.67 |
| SVM | 0.55 | 0.55 | 0.55 | 0.54 | 0.58 |
| AdaBoost | 0.55 | 0.56 | 0.55 | 0.53 | 0.64 |
| XGBoost | 0.55 | 0.55 | 0.55 | 0.54 | 0.58 |
| Decision Tree | 0.54 | 0.54 | 0.54 | 0.54 | 0.53 |
| Logistic Regression | 0.54 | 0.54 | 0.54 | 0.54 | 0.60 |

---

## Summary of Findings
- **Gradient Boosting** achieved the highest accuracy (0.62).  
- **CatBoost** and **Extra Trees** were competitive with strong ROC AUC (~0.66).  
- Simpler models like **Logistic Regression** and **Decision Tree** had lower performance but remain interpretable.  
- Ensemble methods generally outperformed linear models, confirming their advantage for this dataset.
## Additional Resources
[Stepwise Feature Selection Script](src/stepwise_feature_selection.py)
---

## Citation
If you use this repository, please cite:

Acharya, A. B. (2026). *Deciduous Teeth Sex Estimation* [Computer software]. GitHub. https://github.com/ashithacharya/DeciduousTeethSexEstimation
