import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_MODELS = [
    ("logreg", "logreg_model.joblib"),
    ("decisionstump", "decisionstump_model.joblib"),
    ("knn", "knn_model.joblib"),
    ("gaussiannb", "gaussiannb_model.joblib"),
    ("bagging", "bagging_model.joblib"),
    ("randomforest", "randomforest_model.joblib"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect out-of-sample base-model predictions for QBoost. "
            "The script reconstructs the original validation holdout used by the "
            "saved sklearn pipelines, then splits that holdout into QUBO-train, "
            "threshold-tune, and test subsets."
        )
    )
    parser.add_argument("--data", type=str, default="creditcard.csv")
    parser.add_argument("--model-dir", type=str, default=".")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--meta-train-frac",
        type=float,
        default=0.5,
        help="Fraction of the reconstructed holdout reserved for QUBO fitting.",
    )
    parser.add_argument(
        "--tune-frac",
        type=float,
        default=0.5,
        help="Fraction of the remaining holdout reserved for threshold tuning.",
    )
    parser.add_argument(
        "--allow-partial-model-set",
        action="store_true",
        help="Skip incompatible joblib models instead of failing immediately.",
    )
    parser.add_argument("--out", type=str, default="qboost_oof_preds.npz")
    parser.add_argument("--meta-out", type=str, default="qboost_oof_meta.json")
    return parser.parse_args()


def check_fraction(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def timed_predict_proba(model, x_frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = perf_counter()
    probs = model.predict_proba(x_frame)[:, 1]
    elapsed = perf_counter() - start
    return probs.astype(np.float64), float(elapsed)


def collect_split_predictions(
    loaded_models: list[tuple[str, object]],
    x_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    columns = []
    timings: dict[str, float] = {}
    for model_name, model in loaded_models:
        probs, elapsed = timed_predict_proba(model, x_frame)
        columns.append(probs)
        timings[model_name] = elapsed
    matrix = np.column_stack(columns).astype(np.float64)
    return matrix, timings


def load_models(
    model_specs: list[tuple[str, Path]],
    allow_partial_model_set: bool,
) -> tuple[list[tuple[str, object]], list[dict[str, str]]]:
    loaded_models: list[tuple[str, object]] = []
    failures: list[dict[str, str]] = []

    for model_name, model_path in model_specs:
        try:
            model = joblib.load(model_path)
            loaded_models.append((model_name, model))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "name": model_name,
                    "path": str(model_path.resolve()),
                    "error_type": type(exc).__name__,
                    "message": str(exc).splitlines()[0],
                }
            )

    if failures and not allow_partial_model_set:
        lines = [
            "Failed to load one or more pretrained models.",
            "This usually means the joblib files were created with an incompatible scikit-learn version.",
            "Either retrain/re-export those models in the current environment, or rerun with --allow-partial-model-set.",
            "",
            "Failures:",
        ]
        for failure in failures:
            lines.append(
                f"- {failure['name']} ({failure['error_type']}): {failure['message']} [{failure['path']}]"
            )
        raise RuntimeError("\n".join(lines))

    if not loaded_models:
        raise RuntimeError("No compatible models could be loaded.")

    return loaded_models, failures


def summarize_split(y: np.ndarray) -> dict[str, float]:
    return {
        "size": int(y.shape[0]),
        "positive_count": int(y.sum()),
        "positive_rate": float(np.mean(y)),
    }


def main() -> None:
    args = parse_args()
    check_fraction("meta_train_frac", args.meta_train_frac)
    check_fraction("tune_frac", args.tune_frac)

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    meta_path = Path(args.meta_out)

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find dataset: {data_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Cannot find model directory: {model_dir}")

    model_specs = []
    for model_name, rel_path in DEFAULT_MODELS:
        model_path = model_dir / rel_path
        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find model for {model_name}: {model_path}")
        model_specs.append((model_name, model_path))

    loaded_models, failures = load_models(model_specs, args.allow_partial_model_set)

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError(f"Expected a 'Class' column, but got columns: {list(df.columns)}")

    x_all = df.drop(columns=["Class"])
    y_all = df["Class"].astype(int).to_numpy()

    _, x_holdout, _, y_holdout = train_test_split(
        x_all,
        y_all,
        test_size=args.val_size,
        stratify=y_all,
        random_state=args.random_state,
    )

    x_meta_train, x_remaining, y_meta_train, y_remaining = train_test_split(
        x_holdout,
        y_holdout,
        train_size=args.meta_train_frac,
        stratify=y_holdout,
        random_state=args.random_state + 1,
    )

    x_tune, x_test, y_tune, y_test = train_test_split(
        x_remaining,
        y_remaining,
        train_size=args.tune_frac,
        stratify=y_remaining,
        random_state=args.random_state + 2,
    )

    meta_train_probs, meta_train_timings = collect_split_predictions(loaded_models, x_meta_train)
    tune_probs, tune_timings = collect_split_predictions(loaded_models, x_tune)
    test_probs, test_timings = collect_split_predictions(loaded_models, x_test)

    np.savez_compressed(
        out_path,
        model_names=np.asarray([name for name, _ in loaded_models], dtype=object),
        meta_train_probs=meta_train_probs,
        meta_train_y=y_meta_train.astype(np.int8),
        tune_probs=tune_probs,
        tune_y=y_tune.astype(np.int8),
        test_probs=test_probs,
        test_y=y_test.astype(np.int8),
    )

    metadata = {
        "data_path": str(data_path),
        "model_dir": str(model_dir.resolve()),
        "models": [{"name": name, "path": str(path.resolve())} for name, path in model_specs],
        "loaded_model_names": [name for name, _ in loaded_models],
        "load_failures": failures,
        "split_strategy": {
            "reconstructed_base_holdout_size": args.val_size,
            "meta_train_frac_of_holdout": args.meta_train_frac,
            "tune_frac_of_remaining": args.tune_frac,
            "random_state": args.random_state,
        },
        "splits": {
            "meta_train": summarize_split(y_meta_train),
            "tune": summarize_split(y_tune),
            "test": summarize_split(y_test),
        },
        "inference_times_sec": {
            "meta_train": {
                "per_model": meta_train_timings,
                "total": float(sum(meta_train_timings.values())),
            },
            "tune": {
                "per_model": tune_timings,
                "total": float(sum(tune_timings.values())),
            },
            "test": {
                "per_model": test_timings,
                "total": float(sum(test_timings.values())),
            },
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved prediction tensors to: {out_path}")
    print(f"Saved collection metadata to: {meta_path}")
    print("Loaded model order:", ", ".join(name for name, _ in loaded_models))
    if failures:
        print("Skipped incompatible models:", ", ".join(failure["name"] for failure in failures))
    print("Split sizes:", json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
