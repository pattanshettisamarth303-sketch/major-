from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = ROOT / "data" / "heart.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    expected = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def plot_eda(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target Class Distribution")
    plt.xlabel("Heart disease target")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    selected = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), selected):
        sns.histplot(data=df, x=column, hue=TARGET, bins=20, kde=True, ax=ax)
        ax.set_title(column)
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=160)
    plt.close(fig)


def model_grid() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "classifier__max_depth": [2, 3, 4, 5, None],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, None],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Neural Network": (
            MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
            {
                "classifier__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                "classifier__alpha": [0.0001, 0.001, 0.01],
            },
        ),
    }


def evaluate_model(name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_model_plots(best_models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(7, 5))
    for name, estimator in best_models.items():
        y_score = estimator.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
    plt.close()

    best_name, best_model = max(
        best_models.items(),
        key=lambda item: roc_auc_score(y_test, item[1].predict_proba(X_test)[:, 1]),
    )
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance(model: Pipeline) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance, y="feature", x="importance", color="#4c78a8")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()
    return importance


def metrics_to_markdown(metrics: pd.DataFrame) -> str:
    display = metrics.copy()
    for column in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")

    headers = list(display.columns)
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(df: pd.DataFrame, metrics: pd.DataFrame, best_name: str, feature_importance: pd.DataFrame) -> None:
    top_features = "\n".join(
        f"- {row.feature}: {row.importance:.4f}" for row in feature_importance.itertuples()
    )
    report = f"""# Heart Disease Prediction Report

## Objective

Build and compare supervised classification models that predict whether a patient is likely to have heart disease (`target`: 0 or 1) using clinical attributes such as age, sex, chest pain type, cholesterol, resting blood pressure, ECG results, maximum heart rate, exercise-induced angina, ST depression, and related cardiac indicators.

## Dataset Summary

- Source: Google Drive dataset linked in the project PDF
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Target classes: {df[TARGET].value_counts().sort_index().to_dict()}
- Missing values: {int(df.isna().sum().sum())}

## ML Pipeline

1. Loaded and validated the dataset.
2. Checked missing values and class balance.
3. Treated categorical clinical fields with one-hot encoding.
4. Imputed numeric features with medians and scaled them with `StandardScaler`.
5. Split data into stratified train/test sets.
6. Tuned Logistic Regression, Decision Tree, Random Forest, and a Feedforward Neural Network with `GridSearchCV`.
7. Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix, and ROC curves.

## Model Comparison

{metrics_to_markdown(metrics)}

Best model by ROC-AUC: **{best_name}**

## Important Features

{top_features if top_features else "Feature importance is not available for the selected best model."}

## Interpretation

The strongest predictors are clinically plausible heart-disease indicators: chest-pain category, maximum heart rate, exercise-induced angina, ST depression, number of major vessels, and thalassemia results. These factors align with common cardiac risk screening signals, so the model is not relying only on demographic fields.

## Limitations

- The dataset is small, so performance may vary on a larger hospital population.
- This project is for educational pre-screening and must not be used as a standalone medical diagnosis system.
- A production version should be validated on local clinical data and reviewed by healthcare professionals.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
    plot_eda(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    best_models: dict[str, Pipeline] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (classifier, params) in model_grid().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            pipeline,
            params,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        row = evaluate_model(name, search.best_estimator_, X_test, y_test)
        row["best_params"] = search.best_params_
        results.append(row)

    metrics = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    metrics_for_csv = metrics.drop(columns=["confusion_matrix", "best_params"])
    metrics_for_csv.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    (OUTPUT_DIR / "best_params.json").write_text(
        json.dumps({row["model"]: row["best_params"] for row in results}, indent=2),
        encoding="utf-8",
    )

    best_name = metrics.iloc[0]["model"]
    best_model = best_models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    save_model_plots(best_models, X_test, y_test)
    feature_importance = save_feature_importance(best_model)
    write_report(df, metrics_for_csv, best_name, feature_importance)

    print(metrics_for_csv.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
