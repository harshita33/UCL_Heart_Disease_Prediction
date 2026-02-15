# Heart Disease Classification – ML Assignment 2

## Problem Statement
The objective of this project is to predict the presence of heart disease in patients using various machine learning classification algorithms. The problem is formulated as a binary classification task where the target variable indicates whether a patient has heart disease (1) or not (0). The goal is to compare multiple machine learning models based on standard evaluation metrics and identify the best-performing model.

## Dataset Description
The Heart Disease dataset from the UCI Machine Learning Repository is used.
Dataset Characteristics:
Total Instances: 920
Training Samples: 736
Testing Samples: 184
Number of Features: 13
Target Variable: Binary (0 = No Disease, 1 = Disease) = No Disease, 1 = Disease)
The dataset contains categorical and numerical attributes. Categorical variables were encoded into numerical format, and missing values were handled using imputation techniques before model training.

## Models Used and Evaluation Metrics

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1       | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.826087 | 0.903712 | 0.870588  | 0.823529 | 0.846154 | 0.652694 |
| Decision Tree            | 0.755435 | 0.747131 | 0.764706  | 0.823529 | 0.788732 | 0.502158 |
| kNN                      | 0.831522 | 0.901363 | 0.888889  | 0.794118 | 0.839378 | 0.668223 |
| Naive Bayes              | 0.798913 | 0.870397 | 0.823529  | 0.813725 | 0.817734 | 0.593540 |
| Random Forest (Ensemble) | 0.809783 | 0.914694 | 0.813559  | 0.852941 | 0.832536 | 0.613641 |
| XGBoost (Ensemble)       | 0.858696 | 0.899211 | 0.839286  | 0.921569 | 0.878505 | 0.714996 |


## Observations

| ML Model Name            | Observation                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Logistic Regression      | Provided strong baseline performance and good interpretability.                      |
| Decision Tree            | Easy to interpret but slightly prone to overfitting.                                 |
| kNN                      | Performance depends heavily on scaling and choice of k value.                        |
| Naive Bayes              | Fast training but assumes feature independence.                                      |
| Random Forest (Ensemble) | Improved generalization by reducing overfitting through bagging.                     |
| XGBoost (Ensemble)       | Achieved the best overall performance due to boosting and regularization techniques. |

