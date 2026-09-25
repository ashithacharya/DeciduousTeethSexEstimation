# Deciduous Teeth Sex Estimation

## Purpose
This project evaluates whether advanced machine learning algorithms outperform conventional logistic regression in sex estimation using calliper‑derived measurements of the deciduous dentition.

We train and evaluate **10 supervised ML models** (Logistic Regression, Decision Tree, SVM, Gradient Boosting, Random Forest, AdaBoost, XGBoost, LightGBM, Extra Trees, CatBoost) against a logistic regression baseline.  
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

## Model Training
Run the training and evaluation script:

```bash
python src/train_and_evaluate.py

---

#### 3️⃣ Commit your changes
- Scroll down to **Commit changes**.  
- Add a message like:  
- Click **Commit changes**.

---

#### 4️⃣ Verify
Return to your repo’s main page — the README will now display the full table and summary directly below your existing sections.

---

Once you do this, your repository will look complete and professional, with clear documentation of both your workflow and your results.
