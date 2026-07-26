"""
train_models.py
================
Loan Approval Prediction - Model Training Script

This script:
  1. Loads and cleans `loan_approval_dataset.csv`
  2. Encodes categorical columns and scales numeric columns
  3. Splits the data into train/test sets
  4. Trains FOUR classification models:
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Decision Tree
        - Random Forest
  5. Evaluates each model (accuracy, classification report, confusion matrix)
  6. Prints a side-by-side prediction comparison for the test set
  7. Saves a single pickle file (loan_approval_models.pkl) containing every
     trained model + the scaler + the label encoders, so `app.py` (the
     Streamlit deployment) can load everything with one call.

Run:
    python train_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------------------
DATA_PATH = "loan_approval_dataset.csv"
MODEL_OUTPUT_PATH = "loan_approval_models.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that get scaled (continuous numeric features)
SCALE_COLUMNS = [
    "income_annum",
    "loan_amount",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

TARGET_COLUMN = "loan_status"


# --------------------------------------------------------------------------
# 2. LOAD & CLEAN DATA
# --------------------------------------------------------------------------
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Strip whitespace from column names (dataset has leading spaces)
    df.columns = df.columns.str.strip()

    # Strip whitespace from string/object values (e.g. " Graduate" -> "Graduate")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Drop duplicates & the identifier column (not a useful feature)
    df = df.drop_duplicates()
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    return df


# --------------------------------------------------------------------------
# 3. ENCODE + SCALE
# --------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()
    label_encoders = {}

    categorical_cols = ["education", "self_employed", TARGET_COLUMN]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[SCALE_COLUMNS] = scaler.fit_transform(df[SCALE_COLUMNS])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y, scaler, label_encoders


# --------------------------------------------------------------------------
# 4. TRAIN + EVALUATE MODELS
# --------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, target_encoder):

    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }

    trained_models = {}
    predictions = {}
    accuracies = {}

    class_names = target_encoder.classes_  # e.g. ['Approved', 'Rejected']

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        predictions[name] = y_pred
        trained_models[name] = model

        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    return trained_models, predictions, accuracies


# --------------------------------------------------------------------------
# 5. SIDE-BY-SIDE PREDICTION COMPARISON
# --------------------------------------------------------------------------
def show_prediction_comparison(y_test, predictions, target_encoder, n=15):
    comparison = pd.DataFrame({"Actual": target_encoder.inverse_transform(y_test)})

    for name, y_pred in predictions.items():
        comparison[name] = target_encoder.inverse_transform(y_pred)

    print("\n" + "=" * 60)
    print(f"Prediction Comparison (first {n} test samples)")
    print("=" * 60)
    print(comparison.head(n).to_string(index=False))

    comparison.to_csv("predictions_comparison.csv", index=False)
    print("\nFull prediction comparison saved to: predictions_comparison.csv")

    return comparison


def show_accuracy_summary(accuracies):
    summary = pd.DataFrame(
        {"Model": list(accuracies.keys()), "Accuracy": list(accuracies.values())}
    ).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Model Accuracy Summary (sorted best -> worst)")
    print("=" * 60)
    print(summary.to_string(index=False))

    summary.to_csv("model_accuracy_summary.csv", index=False)
    print("\nAccuracy summary saved to: model_accuracy_summary.csv")

    return summary


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------
def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"Dataset shape after cleaning: {df.shape}")

    print("\nPreprocessing (encoding + scaling)...")
    X, y, scaler, label_encoders = preprocess(df)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    target_encoder = label_encoders[TARGET_COLUMN]

    trained_models, predictions, accuracies = train_and_evaluate(
        X_train, X_test, y_train, y_test, target_encoder
    )

    show_prediction_comparison(y_test, predictions, target_encoder)
    show_accuracy_summary(accuracies)

    # ----------------------------------------------------------------
    # Save everything needed for deployment into ONE pickle file
    # ----------------------------------------------------------------
    bundle = {
        "models": trained_models,          # dict: name -> fitted model
        "scaler": scaler,                  # StandardScaler
        "label_encoders": label_encoders,   # dict: col -> LabelEncoder
        "feature_columns": feature_columns,  # exact column order the models expect
        "scale_columns": SCALE_COLUMNS,     # columns the scaler was fit on
        "target_column": TARGET_COLUMN,
        "accuracies": accuracies,
    }

    joblib.dump(bundle, MODEL_OUTPUT_PATH)
    print(f"\nAll models + preprocessing objects saved to: {MODEL_OUTPUT_PATH}")

    # Also save each model individually (handy if you only want one model file)
    for name, model in trained_models.items():
        fname = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, fname)
        print(f"Saved individual model: {fname}")


if __name__ == "__main__":
    main()
