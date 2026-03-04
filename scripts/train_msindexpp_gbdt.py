#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "k",
    "dataset_channels",
    "query_channels",
    "dataset_series_count",
    "avg_series_length",
    "std_series_length",
    "avg_subsequences_per_series",
    "total_subsequences",
    "query_length",
    "query_length_ratio",
    "query_abs_mean_mean",
    "query_std_mean",
    "query_std_std",
    "query_energy_mean",
    "query_energy_std",
]


def load_training_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def build_dataset(rows):
    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.array([int(row["mass_is_better"]) for row in rows], dtype=np.int64)
    return x, y


def export_model(model: GradientBoostingClassifier, output_path: Path):
    y_mean = float(model.init_.class_prior_[1])
    y_mean = min(max(y_mean, 1e-6), 1.0 - 1e-6)
    initial_score = math.log(y_mean / (1.0 - y_mean))

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
        "model_type": "sklearn_gradient_boosting_binary_classifier",
        "feature_names": FEATURE_NAMES,
        "learning_rate": float(model.learning_rate),
        "initial_score": initial_score,
        "trees": trees,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train and export the MSIndexPP GBDT routing model.")
    parser.add_argument("--input", required=True, help="Training CSV produced by Java.")
    parser.add_argument("--output", required=True, help="Output JSON model path.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout ratio for a quick accuracy estimate.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_training_rows(input_path)
    x, y = build_dataset(rows)

    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both MASS and MSINDEX winners.")

    labels, counts = np.unique(y, return_counts=True)
    can_stratify = len(y) > 4 and np.all(counts >= 2)

    if len(y) < 5 or np.any(counts < 2):
        x_train, y_train = x, y
        x_test, y_test = x, y
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y if can_stratify else None,
        )

    model = GradientBoostingClassifier(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    model.fit(x_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(x_train))
    test_accuracy = accuracy_score(y_test, model.predict(x_test))
    export_model(model, output_path)

    print(f"rows={len(rows)}")
    print(f"train_accuracy={train_accuracy:.4f}")
    print(f"test_accuracy={test_accuracy:.4f}")
    print(f"model={output_path}")


if __name__ == "__main__":
    main()
