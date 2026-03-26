import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaBoost using decision stumps.")
    parser.add_argument("--data", type=str, default="creditcard.csv")
    parser.add_argument("--model-out", type=str, default="adaboost_model.joblib")
    parser.add_argument("--metrics-out", type=str, default="adaboost_metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {data_path}")

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError(f"Expected a 'Class' column, but got columns: {list(df.columns)}")

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=args.random_state,
    )

    # AdaBoost supports per-sample weights; for strong class imbalance we weight inversely.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train, clf__sample_weight=sample_weight)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val, y_proba)),
    }

    print("=== Validation metrics (AdaBoost) ===")
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

    print(f"\nSaved model to: {out_model}")
    print(f"Saved metrics to: {out_metrics}")
    print(f"Val positive rate (fraud rate): {np.mean(y_val):.6f}")


if __name__ == "__main__":
    main()

