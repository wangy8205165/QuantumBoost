import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Logistic Regression for credit card fraud detection.")
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard.csv",
        help="Path to creditcard.csv (default: creditcard.csv in current directory).",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="logreg_model.joblib",
        help="Where to save the trained model pipeline.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="logreg_metrics.json",
        help="Where to save validation metrics (JSON).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation size as a fraction (default: 0.2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {data_path}")

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError(f"Expected a 'Class' column, but got columns: {list(df.columns)}")

    # Features are all columns except label.
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=args.random_state,
    )

    # Pipeline ensures standardization learned on train only.
    # Using class_weight helps with the strong label imbalance.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, y_proba)),
    }

    print("=== Validation metrics (Logistic Regression) ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_val, y_pred))
    print("\n=== Classification report ===")
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))

    out_model = Path(args.model_out)
    out_metrics = Path(args.metrics_out)
    joblib.dump(model, out_model)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Small sanity info.
    print(f"\nSaved model to: {out_model}")
    print(f"Saved metrics to: {out_metrics}")
    print(f"Val positive rate (fraud rate): {np.mean(y_val):.6f}")


if __name__ == "__main__":
    main()

