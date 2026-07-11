# Heart Disease Prediction Using Machine Learning

This project implements the assignment from `Artificial Intelligence_Major Project (1).pdf`.

## What It Contains

- Data cleaning and preprocessing
- Exploratory data analysis plots
- Logistic Regression, Decision Tree, Random Forest, and Feedforward Neural Network models
- Hyperparameter tuning with `GridSearchCV`
- Evaluation with Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix, and ROC curves
- Feature-importance interpretation
- A simple Streamlit prediction interface

## Project Structure

```text
data/heart.csv                 Dataset from the PDF link
src/train_models.py            Training, tuning, EDA, evaluation, and report script
outputs/                       Generated metrics, plots, and report
models/best_model.joblib       Saved best model after training
app.py                         Streamlit prediction app
requirements.txt               Python dependencies
```

## How To Run

```powershell
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\streamlit.exe run app.py
```

If `.venv` is not available, create it and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Outputs

After training, see:

- `outputs/report.md`
- `outputs/model_metrics.csv`
- `outputs/best_params.json`
- `outputs/roc_curves.png`
- `outputs/best_confusion_matrix.png`
- `outputs/feature_importance.png`
- `outputs/target_distribution.png`
- `outputs/correlation_heatmap.png`
- `outputs/numeric_distributions.png`

## Note

This model is for educational use and pre-screening demonstration only. It is not a medical diagnosis tool.
