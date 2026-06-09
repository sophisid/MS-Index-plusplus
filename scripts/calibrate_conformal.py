#!/usr/bin/env python3
"""
calibrate_conformal.py

Adds a conformal threshold to an already-trained gbdt-model.json so that
early termination comes with a probabilistic accuracy guarantee.

Given a desired failure rate delta (default 0.1), the calibrated threshold τ
satisfies on the held-out calibration split:

    P(achieved_accuracy >= accuracy_target | model_score >= τ) >= 1 - delta

The threshold is written back into the model JSON as "conformal_threshold"
and "coverage_target".  The Java side (PortableGBDTModel) reads these fields
and uses conformal_threshold in place of the fixed msIndexPPModelScoreThreshold
parameter when present.

Usage:
    python3 scripts/calibrate_conformal.py \\
        --model  artifacts/msindexpp_suite/<dataset>/gbdt-model.json \\
        --data   artifacts/msindexpp_suite/<dataset>/training-data.jsonl \\
        [--delta 0.1] [--cal-size 0.2] [--random-state 42]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# ── feature list (must match TraversalFeatureSnapshot.FEATURE_NAMES) ─────────
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


# ── portable GBDT inference (mirrors PortableGBDTModel.java) ─────────────────
def _tree_predict(tree, feature_map):
    node = 0
    cl, cr, feat, thr, val = (
        tree["children_left"],
        tree["children_right"],
        tree["feature"],
        tree["threshold"],
        tree["value"],
    )
    while cl[node] != -1:
        fi = feat[node]
        fv = feature_map[fi] if 0 <= fi < len(feature_map) else 0.0
        node = cl[node] if fv <= thr[node] else cr[node]
    return val[node]


def predict_proba(model_payload, x):
    """Score rows in x using the portable GBDT payload. Returns P(label=1)."""
    feature_names = model_payload["feature_names"]
    lr   = model_payload["learning_rate"]
    init = model_payload["init_score"]
    trees = model_payload["trees"]

    # Align columns to the model's feature order
    col_idx = [FEATURE_NAMES.index(f) if f in FEATURE_NAMES else -1
               for f in feature_names]

    scores = np.full(len(x), init)
    for tree in trees:
        for i, row in enumerate(x):
            fmap = [row[ci] if ci >= 0 else 0.0 for ci in col_idx]
            scores[i] += lr * _tree_predict(tree, fmap)
    return 1.0 / (1.0 + np.exp(-scores))


# ── data helpers ──────────────────────────────────────────────────────────────
def load_rows(path: Path):
    rows = []
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "label" in obj:
                    rows.append(obj)
            except json.JSONDecodeError:
                scan = 0
                while scan < len(line):
                    start = line.find("{", scan)
                    if start == -1:
                        break
                    try:
                        obj, end = decoder.raw_decode(line, start)
                        if isinstance(obj, dict) and "label" in obj:
                            rows.append(obj)
                        scan = end
                    except json.JSONDecodeError:
                        scan = start + 1
    return rows


def build_arrays(rows):
    x = np.array([[float(r.get(f, 0.0)) for f in FEATURE_NAMES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    g = np.array([int(r.get("query_id", -1)) for r in rows])
    return x, y, g


def cal_test_split(x, y, g, cal_size, test_size, random_state):
    """Split into calibration and test sets, keeping query groups together."""
    unique_g = np.unique(g)
    use_groups = len(unique_g) > 1 and np.all(g >= 0)
    total_holdout = cal_size + test_size

    if use_groups:
        sp1 = GroupShuffleSplit(1, test_size=total_holdout, random_state=random_state)
        _, holdout_idx = next(sp1.split(x, y, groups=g))
        cal_frac = cal_size / total_holdout
        sp2 = GroupShuffleSplit(1, test_size=1 - cal_frac, random_state=random_state + 1)
        cal_rel, test_rel = next(sp2.split(x[holdout_idx], y[holdout_idx], groups=g[holdout_idx]))
        return holdout_idx[cal_rel], holdout_idx[test_rel]
    else:
        idx = np.arange(len(x))
        _, holdout_idx = train_test_split(idx, test_size=total_holdout,
                                          random_state=random_state, stratify=y)
        cal_frac = cal_size / total_holdout
        cal_idx, test_idx = train_test_split(holdout_idx, test_size=1 - cal_frac,
                                             random_state=random_state + 1,
                                             stratify=y[holdout_idx])
        return cal_idx, test_idx


# ── calibration ───────────────────────────────────────────────────────────────
def find_conformal_threshold(scores_cal, y_cal, delta, min_samples=5):
    """
    Return the smallest score threshold τ such that
    precision@τ = P(label=1 | score >= τ) >= 1-delta on the calibration set.
    Falls back to 1.0 (never stop) if no threshold satisfies the constraint.
    """
    candidates = np.sort(np.unique(scores_cal))   # ascending
    best = 1.0
    for tau in candidates:
        mask = scores_cal >= tau
        if mask.sum() < min_samples:
            continue
        if float(y_cal[mask].mean()) >= 1.0 - delta:
            best = float(tau)
            break   # smallest tau that satisfies coverage — done
    return best


def coverage_stats(scores, y, tau):
    mask = scores >= tau
    n = int(mask.sum())
    prec = float(y[mask].mean()) if n > 0 else float("nan")
    return prec, n


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to gbdt-model.json (will be updated in-place).")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to training-data.jsonl used to fit the model.")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Allowed failure rate. Guarantee: "
                             "P(accuracy >= target | stop) >= 1-delta. Default: 0.1")
    parser.add_argument("--cal-size", type=float, default=0.2,
                        help="Fraction of queries held out for calibration.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of queries held out for test (reporting only).")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    model_payload = json.loads(args.model.read_text(encoding="utf-8"))
    if model_payload.get("constant_probability") is not None:
        print("Model is a constant-probability model — conformal calibration not applicable.")
        return

    print("Loading training data...")
    rows = load_rows(args.data)
    x, y, g = build_arrays(rows)
    print(f"  {len(rows)} rows from {len(np.unique(g[g >= 0]))} queries")

    cal_idx, test_idx = cal_test_split(x, y, g, args.cal_size, args.test_size, args.random_state)
    x_cal, y_cal = x[cal_idx], y[cal_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    print(f"  Cal rows: {len(cal_idx)}   Test rows: {len(test_idx)}")

    print("Scoring calibration and test sets...")
    scores_cal  = predict_proba(model_payload, x_cal)
    scores_test = predict_proba(model_payload, x_test)

    tau = find_conformal_threshold(scores_cal, y_cal, args.delta)
    cal_prec,  cal_stops  = coverage_stats(scores_cal,  y_cal,  tau)
    test_prec, test_stops = coverage_stats(scores_test, y_test, tau)

    print(f"\n--- Conformal calibration (delta={args.delta}, target coverage={1-args.delta:.0%}) ---")
    print(f"  Calibrated threshold  : {tau:.6f}")
    print(f"  Coverage on cal  set  : {cal_prec:.4f}  ({cal_stops} stops / {len(cal_idx)} rows)")
    print(f"  Coverage on test set  : {test_prec:.4f}  ({test_stops} stops / {len(test_idx)} rows)")

    model_payload["conformal_threshold"] = tau
    model_payload["coverage_target"]     = round(1.0 - args.delta, 10)
    args.model.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    print(f"\nUpdated {args.model}")


if __name__ == "__main__":
    main()
