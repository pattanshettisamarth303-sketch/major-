# Heart Disease Prediction Report

## Objective

Build and compare supervised classification models that predict whether a patient is likely to have heart disease (`target`: 0 or 1) using clinical attributes such as age, sex, chest pain type, cholesterol, resting blood pressure, ECG results, maximum heart rate, exercise-induced angina, ST depression, and related cardiac indicators.

## Dataset Summary

- Source: Google Drive dataset linked in the project PDF
- Rows: 303
- Columns: 14
- Target classes: {0: 138, 1: 165}
- Missing values: 0

## ML Pipeline

1. Loaded and validated the dataset.
2. Checked missing values and class balance.
3. Treated categorical clinical fields with one-hot encoding.
4. Imputed numeric features with medians and scaled them with `StandardScaler`.
5. Split data into stratified train/test sets.
6. Tuned Logistic Regression, Decision Tree, Random Forest, and a Feedforward Neural Network with `GridSearchCV`.
7. Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix, and ROC curves.

## Model Comparison

| model | accuracy | precision | recall | f1_score | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.820 | 0.775 | 0.939 | 0.849 | 0.909 |
| Logistic Regression | 0.852 | 0.853 | 0.879 | 0.866 | 0.904 |
| Neural Network | 0.803 | 0.800 | 0.848 | 0.824 | 0.865 |
| Decision Tree | 0.820 | 0.806 | 0.879 | 0.841 | 0.813 |

Best model by ROC-AUC: **Random Forest**

## Important Features

- categorical__cp_0: 0.1376
- categorical__thal_2: 0.1321
- categorical__thal_3: 0.0837
- numeric__thalach: 0.0829
- numeric__oldpeak: 0.0827
- categorical__ca_0: 0.0797
- categorical__exang_1: 0.0565
- categorical__exang_0: 0.0491
- categorical__slope_2: 0.0460
- numeric__age: 0.0382
- numeric__chol: 0.0378
- numeric__trestbps: 0.0292
- categorical__slope_1: 0.0254
- categorical__cp_2: 0.0227
- categorical__sex_1: 0.0180

## Interpretation

The strongest predictors are clinically plausible heart-disease indicators: chest-pain category, maximum heart rate, exercise-induced angina, ST depression, number of major vessels, and thalassemia results. These factors align with common cardiac risk screening signals, so the model is not relying only on demographic fields.

## Limitations

- The dataset is small, so performance may vary on a larger hospital population.
- This project is for educational pre-screening and must not be used as a standalone medical diagnosis system.
- A production version should be validated on local clinical data and reviewed by healthcare professionals.
