<div align="center">

# Machine Learning Internship Projects

**Four end-to-end ML projects: classification, regression, imbalanced learning and NLP**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4)
![NLTK](https://img.shields.io/badge/NLTK-NLP-2E8B57)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

</div>

---

## Overview

This repository contains four self-contained Jupyter notebooks. Each one covers a complete workflow: loading data, cleaning, feature engineering, modeling, evaluation and visualization.

| # | Project | Problem type | Models | Best result |
|---|---------|--------------|--------|-------------|
| 1 | [Iris Classification](#task-1--multi-class-classification-iris) | Multi-class classification | Logistic Regression, Decision Tree | **93.3%** accuracy |
| 2 | [Housing Price Prediction](#task-2--housing-price-prediction-california) | Regression | OLS, Ridge, Lasso | **R² = 0.608**, RMSE ≈ 71.7K USD |
| 3 | [Customer Churn Prediction](#task-3--customer-churn-prediction-telco) | Imbalanced binary classification | Random Forest + SMOTE, XGBoost | **ROC-AUC = 0.846** |
| 4 | [Sentiment Analysis](#task-4--nlp-sentiment-classification-tf-idf) | Text classification (NLP) | Multinomial Naive Bayes, Logistic Regression | **83.8%** accuracy |

---

## Repository Structure

```text
ml-internship-projects/
│
├── README.md
├── requirements.txt
├── images/                                  # plots used in this README
│
├── task1-iris-classification/
│   ├── iris_classification_pipeline.ipynb
│   └── iris.csv
│
├── task2-housing-regression/
│   ├── housing_regression_ridge_lasso.ipynb
│   └── housing.csv
│
├── task3-churn-prediction/
│   ├── churn_prediction_rf_xgboost.ipynb
│   └── telco_churn.csv
│
└── task4-sentiment-analysis/
    ├── sentiment_tfidf_naive_bayes.ipynb
    └── sentiment_reviews.csv
```

---

## Task 1 | Multi-Class Classification (Iris)

**Goal:** Classify iris flowers into 3 species and benchmark two models.

**Approach**
- Exploratory pairplots with Seaborn
- Feature scaling with `StandardScaler`, stratified 80/20 train/test split
- Logistic Regression and Decision Tree
- Metrics: accuracy, precision, recall, F1-score, confusion matrix heatmaps, decision boundary plots

**Results (30 test samples)**

| Model | Accuracy | Precision | Recall | F1-score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

<p align="center"><img src="images/task1_confusion_matrix.png" width="650" alt="Iris confusion matrices"></p>

**Key insight:** *Setosa* is separated perfectly. All errors happen between *versicolor* and *virginica*, which overlap in feature space.

---

## Task 2 | Housing Price Prediction (California)

**Goal:** Predict median house value and compare OLS, Ridge (L2) and Lasso (L1) regression.

**Approach**
- Median imputation for 207 missing `total_bedrooms` values (fit on training data only)
- Correlation heatmap and **VIF analysis** to detect multicollinearity
- Engineered features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household` and a centered `income × rooms` interaction. Raw count columns were dropped, which cut their VIF from 6–28 down to about 1–3 (`latitude` and `longitude` stay near 9)
- `StandardScaler` for numeric columns and One-Hot encoding for `ocean_proximity`
- `GridSearchCV` (5-fold) to tune `alpha` for Ridge and Lasso

**Results (test set)**

| Model | RMSE | MAE | R² |
|-------|-----:|----:|:--:|
| OLS | 71,651 | 51,568 | 0.608 |
| Ridge (L2) | 71,654 | 51,571 | 0.608 |
| Lasso (L1) | 71,652 | 51,570 | 0.608 |

<p align="center"><img src="images/task2_predicted_vs_actual.png" width="850" alt="Predicted vs actual house values"></p>

**Key insight:** The three models perform almost identically, because the dataset is large (20,640 rows) with few features, so regularization has little to fix. The vertical line at the right comes from the dataset's price cap of 500,001. A non-linear model (e.g. Gradient Boosting) would likely perform better.

---

## Task 3 | Customer Churn Prediction (Telco)

**Goal:** Predict which telecom customers will leave, and reach **ROC-AUC above 0.80**.

**Approach**
- Converted `TotalCharges` to numeric. The 11 blank values belong to customers with `tenure = 0`, so they were set to 0
- `OneHotEncoder` for categorical features, `StandardScaler` for numeric features
- Class imbalance (26.5% churn):
  - Random Forest with **SMOTE**, applied inside an `imblearn` pipeline so synthetic samples are created only from training data
  - XGBoost with **`scale_pos_weight`**
- 5-fold stratified cross-validation, ROC curves, confusion matrices and top-10 feature importances

**Results (test set, threshold 0.5)**

| Model | ROC-AUC | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|-------|:-------:|:--------:|:-----------------:|:--------------:|:----------:|
| Random Forest + SMOTE | 0.834 | 0.767 | 0.550 | 0.660 | 0.600 |
| **XGBoost + class weight** | **0.846** | 0.752 | 0.522 | **0.789** | **0.628** |

<p align="center">
  <img src="images/task3_roc_curve.png" width="380" alt="ROC curves">
  <img src="images/task3_feature_importance.png" width="560" alt="Top 10 feature importances">
</p>

**Key insight:** Customers on **month-to-month contracts** with **short tenure**, **no online security** or **no tech support** are the most likely to churn. XGBoost catches about 79% of churners.

---

## Task 4 | NLP Sentiment Classification (TF-IDF)

**Goal:** Classify review text as positive or negative.

**Approach**
- Text cleaning: lowercase, HTML tag removal, punctuation removal, stopword removal and lemmatization (NLTK). Negation words such as *not* and *never* are kept because they change sentiment
- `TfidfVectorizer(max_features=5000)` with unigrams and bigrams
- Multinomial Naive Bayes and Logistic Regression
- `predict_sentiment()` helper to test any custom phrase

**Data note:** The provided `sentiment_reviews.csv` has only 10 reviews, which is too small for a reliable accuracy score. The notebook runs the full pipeline on it, then benchmarks accuracy on the **NLTK `movie_reviews` corpus** (2,000 reviews).

**Results (400 test reviews)**

| Model | 5-fold CV accuracy | Test accuracy |
|-------|:------------------:|:-------------:|
| Multinomial Naive Bayes | 0.820 | 0.815 |
| **Logistic Regression** | **0.830** | **0.838** |

<p align="center"><img src="images/task4_model_comparison.png" width="850" alt="Sentiment model comparison"></p>

**Key insight and limitation:** The model learned from *movie* reviews, so it transfers poorly to product reviews. On the 9 provided product reviews it scores only 5 out of 9. Training on a large product review dataset, or using a pretrained transformer, would be the next improvement.

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/vamsidhar-reddy-81/ml-internship-projects.git
cd ml-internship-projects
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run a notebook**
```bash
cd task3-churn-prediction
jupyter notebook
```
Open the `.ipynb` file and choose **Kernel → Restart & Run All**.

> Each notebook reads its CSV from the same folder, so run it from inside its own task folder.
> Task 4 downloads NLTK data on the first run, so an internet connection is needed.

---

## Datasets

| Task | Dataset | Source |
|------|---------|--------|
| 1 | Iris (150 samples, 3 classes) | Provided with the assignment |
| 2 | California Housing (20,640 rows) | [ageron/handson-ml2](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) |
| 3 | Telco Customer Churn (7,043 rows) | [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) |
| 4 | Sentiment reviews (10 rows) + NLTK `movie_reviews` (2,000 rows) | Provided with the assignment and [NLTK](https://www.nltk.org/nltk_data/) |

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-Learn` · `XGBoost` · `imbalanced-learn` · `NLTK` · `Matplotlib` · `Seaborn` · `Jupyter`

---

## Author

**Morreddy Vamsidhar Reddy**
B.Tech (AI & ML), Tirumala Engineering College

[GitHub](https://github.com/vamsidhar-reddy-81) · [Portfolio](https://vamsidhar-reddy-81.github.io/VamsidharReddy-Portfolio/)
