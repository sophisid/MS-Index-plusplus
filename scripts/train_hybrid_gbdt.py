#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "k",
    "avg_series_length",
    "std_series_length",
    "total_subsequences",
    "query_length_ratio",
    "query_mu_mean",
    "query_sigma2_mean",
    "query_abs_mean_mean",
    "query_std_mean",
    "query_std_std",
    "query_energy_mean",
    "query_energy_std",
]


def load_rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("No training rows found in {}".format(path))

    expected_header = ["query_id"] + FEATURE_NAMES + [
        "mass_time_ms",
        "msindex_time_ms",
        "better_algorithm",
        "mass_is_better",
    ]

    first_values = lines[0].split(",")
    has_header = first_values and first_values[0] == "query_id"
    if has_header:
        if len(lines) <= 1:
            raise ValueError("No training rows found in {}".format(path))
        header = first_values
        data_lines = lines[1:]
    else:
        header = expected_header
        data_lines = lines

    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    if not rows:
        raise ValueError("No training rows found in {}".format(path))
    return rows


def build_dataset(rows):
    x = np.array([[float(row[feature]) for feature in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.array([int(row["mass_is_better"]) for row in rows], dtype=int)
    return x, y


def compute_sample_weights(y):
    counts = Counter(int(label) for label in y.tolist())
    n_classes = len(counts)
    n_samples = len(y)
    class_weights = {
        label: n_samples / (n_classes * count)
        for label, count in counts.items()
        if count > 0
    }
    return np.array([class_weights[int(label)] for label in y], dtype=float)


def get_init_score(model):
    if hasattr(model, "_raw_predict_init") and hasattr(model, "n_features_in_"):
        sample = np.zeros((1, model.n_features_in_), dtype=float)
        return float(np.ravel(model._raw_predict_init(sample))[0])

    init_estimator = model.init_
    if hasattr(init_estimator, "constant_"):
        return float(np.ravel(init_estimator.constant_)[0])

    if hasattr(init_estimator, "class_prior_"):
        positive_probability = float(np.ravel(init_estimator.class_prior_)[-1])
        positive_probability = float(np.clip(positive_probability, 1e-12, 1 - 1e-12))
        return float(np.log(positive_probability / (1.0 - positive_probability)))

    raise AttributeError("Could not extract GradientBoostingClassifier initial score")


def export_model(model, output_path: Path, positive_probability: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trees = []
    for estimator in model.estimators_.ravel():
        tree = estimator.tree_
        trees.append(
            {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value[:, 0, 0].tolist(),
            }
        )

    payload = {
        "feature_names": FEATURE_NAMES,
        "learning_rate": float(model.learning_rate),
        "init_score": get_init_score(model) if positive_probability is None else float(
            np.log(
                np.clip(positive_probability, 1e-12, 1 - 1e-12) /
                (1.0 - np.clip(positive_probability, 1e-12, 1 - 1e-12))
            )
        ),
        "trees": trees,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_constant_model(output_path: Path, positive_probability: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": FEATURE_NAMES,
        "learning_rate": 0.0,
        "init_score": 0.0,
        "constant_probability": float(positive_probability),
        "trees": [],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            from sklearn.cross_validation import train_test_split
    except ImportError as exc:
        raise SystemExit("scikit-learn is required to train the hybrid model: {}".format(exc))

    parser = argparse.ArgumentParser(description="Train a portable GBDT model for HYBRID (MASS vs MSINDEX).")
    parser.add_argument("--input", required=True, type=Path, help="Path to the hybrid training CSV exported by Java.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the exported portable JSON model.")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    x, y = build_dataset(rows)
    counts = Counter(int(label) for label in y.tolist())
    print("Label distribution: MSINDEX=0 -> {}, MASS=1 -> {}".format(counts.get(0, 0), counts.get(1, 0)))

    if len(counts) < 2:
        constant_probability = float(y[0]) if len(y) else 0.0
        print("Only one class present. Exporting constant model with probability {:.1f}".format(constant_probability))
        export_constant_model(args.output, constant_probability)
        return

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    sample_weight = compute_sample_weights(y_train)

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        random_state=args.random_state,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    try:
        print("ROC AUC: {:.4f}".format(roc_auc_score(y_test, y_proba)))
    except ValueError:
        print("ROC AUC: unavailable")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["MSINDEX", "MASS"], zero_division=0))

    export_model(model, args.output, float(np.average(y_train, weights=sample_weight)))
    print("Saved portable model to {}".format(args.output))


if __name__ == "__main__":
    main()
