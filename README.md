# Customer Churn Prediction

This project was developed as part of a personal machine learning portfolio to practice the complete workflow of a binary classification problem, from exploratory data analysis to model evaluation.

## Project Overview

This project focuses on predicting customer churn for a telecom company using machine learning models.  
The goal is to identify customers who are more likely to leave the service and to extract interpretable business insights from the data.


## Dataset

The project uses the Telco Customer Churn dataset from Kaggle.

Dataset characteristics:
- 7043 customers
- 20 raw features
- 30 engineered features
- Binary target: Churn / No Churn

The dataset should be placed in `data/raw/Telco-Customer_Churn.csv`.

## Exploratory Data Analysis

Exploratory Data Analysis revealed several relevant churn patterns:

- Customers with short tenure are more likely to churn.
- Month-to-month contracts are strongly associated with churn.
- Fiber optic customers show higher churn rates.
- Electronic check payment is associated with higher churn.
- Customers with lower cumulative charges but higher monthly charges are more likely to churn, suggesting that churn is concentrated among relatively recent customers.

These findings suggest that churn is strongly influenced by customer tenure, contract type, payment method, and service characteristics.

## Data Preprocessing

The preprocessing pipeline includes:

- Cleaning column names.
- Converting `TotalCharges` to numeric format.
- Filling missing `TotalCharges` values with 0, since all missing values correspond to customers with tenure equal to 0.
- Encoding the binary target `Churn` as 0/1.
- Mapping binary categorical features to 0/1.
- One-hot encoding multi-category features.
- Dropping `customerID`, since it is an identifier and does not contain predictive information.

## Models

Two machine learning models were trained and compared:

### Logistic Regression

Logistic Regression was used as an interpretable baseline model.
Since it is sensitive to feature scaling, numerical features were standardized before training.

### Random Forest

A Random Forest classifier was also trained.
After evaluating a baseline Random Forest, a tuned version was trained with the following hyperparameters::

```python
n_estimators=500
max_depth=10
min_samples_leaf=5
```

## Results

Logistic Regression provided a strong baseline and achieved slightly better recall for churn customers.
The tuned Random Forest achieved a slightly higher ROC-AUC and precision, but lower recall on the churn class.

Overall, the results suggest that the dataset contains a few highly informative features and that a simple linear model can already perform competitively when the data is properly cleaned and encoded.

Running `train.py` automatically generates a `results/model_comparison.csv` file containing the evaluation metrics for all trained models.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|---------:|----------:|-------:|-----:|--------:|
| Logistic Regression | 0.807 | 0.66 | 0.57 | 0.61 | 0.842 |
| Tuned Random Forest | 0.807 | 0.68 | 0.52 | 0.59 | 0.845 |

## Business Insights

The analysis suggests that the company should pay particular attention to:

- New customers with low tenure.
- Customers with month-to-month contracts.
- Customers using electronic check as payment method.
- Fiber optic customers with high monthly charges.

These customer segments could be targeted with retention strategies, such as contract incentives, improved onboarding, or personalized offers.

## Repository Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── results/
│    └── model_comparison.csv
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   ├── compare.py
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone the repository.
2. Download the dataset from Kaggle, and place it inside `data/raw/`.
3. Install the required packages:

```bash
pip install -r requirements.txt
```
4. Run the training script from the project root:

```bash
python src/train.py
```

## Future Improvements

Possible future improvements include:

- Threshold optimization to improve churn recall.
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Additional feature engineering.
- Saving trained models for later inference.
- Deployment as a simple API or dashboard.