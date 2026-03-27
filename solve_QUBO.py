import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the QBoost QUBO using simulated annealing. "
            "Weights are represented with R binary precision bits per weak learner."
        )
    )
    parser.add_argument("--preds", type=str, default="qboost_oof_preds.npz")
    parser.add_argument("--out", type=str, default="qboost_weights.json")
    parser.add_argument("--r-bits", type=int, default=4)
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=1e-3,
        help="L2 regularization on ensemble weights.",
    )
    parser.add_argument(
        "--simplex-penalty",
        type=float,
        default=10.0,
        help="Penalty for violating sum(weights)=1.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=-1.0,
        help="Positive-class sample weight. Use a negative value to auto-set from class imbalance.",
    )
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--num-reads", type=int, default=128)
    parser.add_argument("--num-sweeps", type=int, default=400)
    parser.add_argument("--temp-start", type=float, default=3.0)
    parser.add_argument("--temp-end", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_bit_matrix(num_models: int, r_bits: int) -> np.ndarray:
    bit_values = np.asarray([2.0 ** (-r) for r in range(r_bits)], dtype=np.float64)
    basis = np.zeros((num_models, num_models * r_bits), dtype=np.float64)
    for model_idx in range(num_models):
        start = model_idx * r_bits
        basis[model_idx, start : start + r_bits] = bit_values
    return basis


def build_qubo(
    h_matrix: np.ndarray,
    y: np.ndarray,
    r_bits: int,
    lambda_reg: float,
    simplex_penalty: float,
    pos_weight: float,
    neg_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    num_samples, num_models = h_matrix.shape
    basis = build_bit_matrix(num_models, r_bits)
    x_bits = h_matrix @ basis

    if pos_weight < 0.0:
        positive_count = max(int(np.sum(y == 1)), 1)
        negative_count = max(int(np.sum(y == 0)), 1)
        pos_weight = negative_count / positive_count

    sample_weights = np.where(y == 1, pos_weight, neg_weight).astype(np.float64)
    weighted_x = x_bits * sample_weights[:, None]
    quadratic = (x_bits.T @ weighted_x) / num_samples
    linear = (-2.0 * ((sample_weights * y) @ x_bits)) / num_samples

    reg_quadratic = lambda_reg * (basis.T @ basis)
    simplex_vector = basis.sum(axis=0)
    simplex_quadratic = simplex_penalty * np.outer(simplex_vector, simplex_vector)
    simplex_linear = -2.0 * simplex_penalty * simplex_vector

    qubo_quadratic = quadratic + reg_quadratic + simplex_quadratic
    qubo_linear = linear + simplex_linear
    return qubo_quadratic, qubo_linear, basis, sample_weights, float(pos_weight)


def energy(state: np.ndarray, quadratic: np.ndarray, linear: np.ndarray) -> float:
    return float(state @ quadratic @ state + linear @ state)


def simulated_annealing(
    quadratic: np.ndarray,
    linear: np.ndarray,
    num_reads: int,
    num_sweeps: int,
    temp_start: float,
    temp_end: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    num_bits = linear.shape[0]
    temperatures = np.geomspace(temp_start, temp_end, num_sweeps)

    best_state = None
    best_energy = None

    for _ in range(num_reads):
        state = rng.integers(0, 2, size=num_bits, dtype=np.int8).astype(np.float64)
        current_energy = energy(state, quadratic, linear)

        if best_energy is None or current_energy < best_energy:
            best_state = state.copy()
            best_energy = current_energy

        for temperature in temperatures:
            for bit_idx in rng.permutation(num_bits):
                candidate = state.copy()
                candidate[bit_idx] = 1.0 - candidate[bit_idx]
                candidate_energy = energy(candidate, quadratic, linear)
                delta = candidate_energy - current_energy

                if delta <= 0.0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                    state = candidate
                    current_energy = candidate_energy
                    if current_energy < best_energy:
                        best_state = state.copy()
                        best_energy = current_energy

    return best_state.astype(np.int8), float(best_energy)


def compute_objective(
    h_matrix: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_reg: float,
    simplex_penalty: float,
    sample_weights: np.ndarray,
) -> dict[str, float]:
    residual = h_matrix @ weights - y
    weighted_mse = float(np.mean(sample_weights * residual**2))
    l2_penalty = float(lambda_reg * np.sum(weights**2))
    simplex_error = float(simplex_penalty * (np.sum(weights) - 1.0) ** 2)
    return {
        "weighted_mse": weighted_mse,
        "l2_penalty": l2_penalty,
        "simplex_penalty": simplex_error,
        "total": weighted_mse + l2_penalty + simplex_error,
    }


def main() -> None:
    args = parse_args()
    preds_path = Path(args.preds)
    out_path = Path(args.out)

    if args.r_bits <= 0:
        raise ValueError(f"r_bits must be positive, got {args.r_bits}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Cannot find prediction matrix file: {preds_path}")

    data = np.load(preds_path, allow_pickle=True)
    model_names = [str(name) for name in data["model_names"].tolist()]
    h_train = np.asarray(data["meta_train_probs"], dtype=np.float64)
    y_train = np.asarray(data["meta_train_y"], dtype=np.float64)

    quadratic, linear, basis, sample_weights, resolved_pos_weight = build_qubo(
        h_matrix=h_train,
        y=y_train,
        r_bits=args.r_bits,
        lambda_reg=args.lambda_reg,
        simplex_penalty=args.simplex_penalty,
        pos_weight=args.pos_weight,
        neg_weight=args.neg_weight,
    )

    train_start = perf_counter()
    best_bits, best_energy = simulated_annealing(
        quadratic=quadratic,
        linear=linear,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        seed=args.seed,
    )
    training_time = perf_counter() - train_start

    raw_weights = basis @ best_bits.astype(np.float64)
    raw_weight_sum = float(np.sum(raw_weights))
    if raw_weight_sum <= 0.0:
        normalized_weights = np.full_like(raw_weights, 1.0 / len(raw_weights))
    else:
        normalized_weights = raw_weights / raw_weight_sum

    objective_raw = compute_objective(
        h_matrix=h_train,
        y=y_train,
        weights=raw_weights,
        lambda_reg=args.lambda_reg,
        simplex_penalty=args.simplex_penalty,
        sample_weights=sample_weights,
    )
    objective_normalized = compute_objective(
        h_matrix=h_train,
        y=y_train,
        weights=normalized_weights,
        lambda_reg=args.lambda_reg,
        simplex_penalty=args.simplex_penalty,
        sample_weights=sample_weights,
    )

    result = {
        "model_names": model_names,
        "r_bits": args.r_bits,
        "bit_values": [float(2.0 ** (-r)) for r in range(args.r_bits)],
        "bitstring": "".join(str(int(bit)) for bit in best_bits.tolist()),
        "solver": {
            "name": "custom_simulated_annealing",
            "num_reads": args.num_reads,
            "num_sweeps": args.num_sweeps,
            "temp_start": args.temp_start,
            "temp_end": args.temp_end,
            "seed": args.seed,
        },
        "objective_setup": {
            "lambda_reg": args.lambda_reg,
            "simplex_penalty": args.simplex_penalty,
            "neg_weight": args.neg_weight,
            "pos_weight": resolved_pos_weight,
        },
        "training_time_sec": float(training_time),
        "best_energy": float(best_energy),
        "raw_weights": {name: float(weight) for name, weight in zip(model_names, raw_weights.tolist())},
        "normalized_weights": {
            name: float(weight) for name, weight in zip(model_names, normalized_weights.tolist())
        },
        "sum_raw_weights": raw_weight_sum,
        "train_objective_raw": objective_raw,
        "train_objective_normalized": objective_normalized,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved QBoost weights to: {out_path}")
    print("Normalized weights:", json.dumps(result["normalized_weights"], indent=2))
    print(f"Training time (sec): {training_time:.6f}")
    print(f"Best energy: {best_energy:.6f}")


if __name__ == "__main__":
    main()
