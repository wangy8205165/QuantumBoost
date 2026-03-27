import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune the decision threshold for QBoost on the tune split and "
            "report final metrics on the held-out test split."
        )
    )
    parser.add_argument("--preds", type=str, default="qboost_oof_preds.npz")
    parser.add_argument("--weights", type=str, default="qboost_weights.json")
    parser.add_argument("--collect-meta", type=str, default="qboost_oof_meta.json")
    parser.add_argument("--out", type=str, default="qboost_eval.json")
    parser.add_argument(
        "--weight-key",
        type=str,
        default="normalized_weights",
        choices=["normalized_weights", "raw_weights"],
    )
    parser.add_argument(
        "--score-repeats",
        type=int,
        default=200,
        help="Number of repeated score computations for a stable inference-time estimate.",
    )
    return parser.parse_args()


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5, 0.0

    f1_values = 2.0 * precisions[:-1] * recalls[:-1] / np.clip(precisions[:-1] + recalls[:-1], 1e-12, None)
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def collect_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(np.int8)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate_pred": float(np.mean(y_pred)),
    }


def timed_scores(h_matrix: np.ndarray, weights: np.ndarray, repeats: int) -> tuple[np.ndarray, float]:
    repeats = max(int(repeats), 1)
    start = perf_counter()
    scores = None
    for _ in range(repeats):
        scores = h_matrix @ weights
    elapsed = perf_counter() - start
    return scores.astype(np.float64), float(elapsed / repeats)


def main() -> None:
    args = parse_args()
    preds_path = Path(args.preds)
    weights_path = Path(args.weights)
    collect_meta_path = Path(args.collect_meta)
    out_path = Path(args.out)

    if not preds_path.exists():
        raise FileNotFoundError(f"Cannot find prediction matrix file: {preds_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Cannot find QBoost weights file: {weights_path}")

    data = np.load(preds_path, allow_pickle=True)
    with weights_path.open("r", encoding="utf-8") as f:
        weights_payload = json.load(f)

    model_names = [str(name) for name in data["model_names"].tolist()]
    weights_dict = weights_payload[args.weight_key]
    weights = np.asarray([weights_dict[name] for name in model_names], dtype=np.float64)

    h_tune = np.asarray(data["tune_probs"], dtype=np.float64)
    y_tune = np.asarray(data["tune_y"], dtype=np.int8)
    h_test = np.asarray(data["test_probs"], dtype=np.float64)
    y_test = np.asarray(data["test_y"], dtype=np.int8)

    tune_scores, tune_avg_inference = timed_scores(h_tune, weights, args.score_repeats)
    best_threshold, tune_best_f1 = select_threshold(y_tune, tune_scores)

    test_scores, test_avg_inference = timed_scores(h_test, weights, args.score_repeats)

    tune_metrics = collect_metrics(y_tune, tune_scores, best_threshold)
    test_metrics = collect_metrics(y_test, test_scores, best_threshold)

    collect_meta = None
    if collect_meta_path.exists():
        with collect_meta_path.open("r", encoding="utf-8") as f:
            collect_meta = json.load(f)

    base_tune_inference = None
    base_test_inference = None
    if collect_meta is not None:
        base_tune_inference = float(collect_meta["inference_times_sec"]["tune"]["total"])
        base_test_inference = float(collect_meta["inference_times_sec"]["test"]["total"])

    result = {
        "model_names": model_names,
        "weight_key": args.weight_key,
        "weights": {name: float(weight) for name, weight in zip(model_names, weights.tolist())},
        "selected_threshold": float(best_threshold),
        "tune_best_f1_during_search": float(tune_best_f1),
        "metrics": {
            "tune": tune_metrics,
            "test": test_metrics,
        },
        "timing_sec": {
            "qubo_training": float(weights_payload["training_time_sec"]),
            "ensemble_inference_avg_tune": float(tune_avg_inference),
            "ensemble_inference_avg_test": float(test_avg_inference),
            "base_model_inference_tune": base_tune_inference,
            "base_model_inference_test": base_test_inference,
            "total_pipeline_inference_test": (
                None if base_test_inference is None else float(base_test_inference + test_avg_inference)
            ),
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved evaluation report to: {out_path}")
    print("Test metrics:", json.dumps(result["metrics"]["test"], indent=2))
    print("Timing:", json.dumps(result["timing_sec"], indent=2))


if __name__ == "__main__":
    main()
