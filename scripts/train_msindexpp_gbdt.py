#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


FEATURE_NAMES = [
    "requested_k",
    "checkpoint_id",
    "distance_computations",
    "candidate_count",
    "unique_series_count",
    "nodes_completed",
    "leafs_completed",
    "non_leaf_expansions",
    "children_enqueued",
    "entries_scanned",
    "frontier_size",
    "frontier_best_distance",
    "frontier_avg_distance",
    "frontier_worst_distance",
    "last_distance",
    "last_candidate_distance",
    "threshold",
    "threshold_minus_frontier_best",
    "candidate_rate",
    "enqueue_per_step",
]


def load_rows(path: Path):
    decoder = json.JSONDecoder()
    rows = []
    recovered_lines = 0
    dropped_fragments = 0
    skipped_lines = 0
    dropped_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "label" in obj:
                        rows.append(obj)
                    else:
                        dropped_rows += 1
                    continue
                except json.JSONDecodeError:
                    # Recover valid JSON objects when a JSONL line is corrupted by
                    # concatenated or truncated records.
                    recovered_on_line = 0
                    scan_idx = 0
                    while scan_idx < len(line):
                        start_idx = line.find("{", scan_idx)
                        if start_idx == -1:
                            break
                        try:
                            obj, end_idx = decoder.raw_decode(line, start_idx)
                        except json.JSONDecodeError:
                            dropped_fragments += 1
                            scan_idx = start_idx + 1
                            continue

                        if isinstance(obj, dict) and "label" in obj:
                            rows.append(obj)
                            recovered_on_line += 1
                        else:
                            dropped_rows += 1
                        scan_idx = end_idx

                    if recovered_on_line == 0:
                        skipped_lines += 1
                        continue
                    recovered_lines += 1
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    if recovered_lines or dropped_fragments or skipped_lines or dropped_rows:
        print(
            f"Recovered rows from {recovered_lines} corrupted line(s); "
            f"dropped {dropped_fragments} malformed fragment(s); "
            f"skipped {skipped_lines} fully corrupted line(s); "
            f"discarded {dropped_rows} incomplete row(s)."
        )
    return rows


def build_dataset(rows):
    x = np.array([[float(row.get(feature, 0.0)) for feature in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    groups = np.array([int(row.get("query_id", -1)) for row in rows], dtype=int)
    return x, y, groups


def describe_labels(y, prefix: str):
    counts = Counter(int(label) for label in y.tolist())
    total = len(y)
    negative = counts.get(0, 0)
    positive = counts.get(1, 0)
    print(
        f"{prefix} label distribution: "
        f"continue=0 -> {negative} ({negative / total:.4%}), "
        f"stop=1 -> {positive} ({positive / total:.4%})"
    )


def split_dataset(x, y, groups, test_size: float, random_state: int):
    unique_groups = np.unique(groups)
    use_group_split = len(unique_groups) > 1 and np.all(groups >= 0)

    if use_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))
        return (
            x[train_idx],
            x[test_idx],
            y[train_idx],
            y[test_idx],
            groups[train_idx],
            groups[test_idx],
            True,
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test, None, None, False


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


def get_init_score(model: GradientBoostingClassifier) -> float:
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


def export_model(model: GradientBoostingClassifier, output_path: Path):
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

    init_score = get_init_score(model)
    payload = {
        "feature_names": FEATURE_NAMES,
        "learning_rate": float(model.learning_rate),
        "init_score": init_score,
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
    parser = argparse.ArgumentParser(description="Train a portable GBDT model for MSIndexPP early termination.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the training JSONL exported by Java.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the exported portable JSON model.")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    x, y, groups = build_dataset(rows)

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        constant_probability = float(unique_labels[0])
        print(f"Rows: {len(rows)}")
        print(f"Only one label present: {int(unique_labels[0])}")
        print(f"Exporting constant model with probability {constant_probability:.1f}")
        export_constant_model(args.output, constant_probability)
        print(f"Model exported to {args.output}")
        return

    x_train, x_test, y_train, y_test, groups_train, groups_test, used_group_split = split_dataset(
        x,
        y,
        groups,
        test_size=args.test_size,
        random_state=args.random_state,
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

    print(f"Rows: {len(rows)}")
    print(f"Unique queries: {len(np.unique(groups[groups >= 0]))}")
    print(f"Split mode: {'grouped by query_id' if used_group_split else 'row-wise stratified'}")
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    if used_group_split:
        print(f"Train queries: {len(np.unique(groups_train))}")
        print(f"Test queries: {len(np.unique(groups_test))}")
    describe_labels(y, "All rows")
    describe_labels(y_train, "Train")
    describe_labels(y_test, "Test")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    export_model(model, args.output)
    print(f"Model exported to {args.output}")


if __name__ == "__main__":
    main()
