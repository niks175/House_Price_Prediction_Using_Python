"""
House Price Prediction - Linear Regression
--------------------------------------------
Trains a linear regression model to predict house prices from a dataset,
evaluates it, and saves the trained model + evaluation plot.

Usage:
    python house_price_prediction.py
    python house_price_prediction.py --input house_price.csv --target price
"""

import argparse
import sys
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train a house price prediction model.")
    parser.add_argument("--input", default="house_price.csv", help="Path to input CSV file")
    parser.add_argument("--target", default="price", help="Name of the target column")
    parser.add_argument("--output-dir", default="output", help="Directory to save model/plots")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    return parser.parse_args()


def load_data(filepath: str, target: str) -> pd.DataFrame:
    """Load dataset and validate the target column exists."""
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Error: file not found -> {filepath}")

    df = pd.read_csv(path)

    if target not in df.columns:
        sys.exit(f"Error: target column '{target}' not found. Available columns: {list(df.columns)}")

    if df[target].isnull().any():
        before = len(df)
        df = df.dropna(subset=[target])
        print(f"Dropped {before - len(df)} row(s) with missing target values.")

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build a preprocessing + model pipeline.
    Numeric columns: impute missing values (median) + scale.
    Categorical columns: impute missing values (most frequent) + one-hot encode.
    Doing this inside a pipeline (fit only on training data) avoids data leakage,
    unlike filling NaNs on the whole dataset before splitting.
    """
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])


def evaluate(y_test, y_pred) -> dict:
    """Compute and print regression evaluation metrics."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance (test set):")
    print(f"  Mean Absolute Error (MAE):  {mae:,.2f}")
    print(f"  Root Mean Squared Error:    {rmse:,.2f}")
    print(f"  R² Score:                   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_predictions(y_test, y_pred, output_dir: Path, show: bool) -> None:
    """Scatter plot of actual vs predicted prices with a perfect-fit reference line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")

    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    plt.xlabel("Actual House Prices")
    plt.ylabel("Predicted House Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    df = load_data(args.input, args.target)
    print("\nDataset Preview:")
    print(df.head())

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipeline = build_pipeline(X)

    # Cross-validation on the training set gives a more reliable estimate
    # of generalization than a single train/test split alone.
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"\nCross-validated R² (train set, 5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    evaluate(y_test, y_pred)
    plot_predictions(y_test, y_pred, output_dir, args.show)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "house_price_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    # Example: predict on the first row of the original feature set
    sample = X.iloc[[0]]
    sample_prediction = pipeline.predict(sample)[0]
    print(f"Predicted price for sample house: {sample_prediction:,.2f}")


if __name__ == "__main__":
    main()
