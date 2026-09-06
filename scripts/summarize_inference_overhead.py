#!/usr/bin/env python3.10
"""Aggregate model-inference counters from Hybrid and MS-Index++ runs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASETS = ("synthetic", "stocks", "wind", "weather", "DuckDuckGeese")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-root", type=Path, default=Path("artifacts_hybrid/results"))
    parser.add_argument("--approx-root", type=Path, default=Path("artifacts_target_independent/target_0p90"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_inference_overhead"))
    return parser.parse_args()


def as_float(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return 0.0


def dataset_for(row, path):
    value = row.get("datasetName") or row.get("dataset")
    if value in DATASETS:
        return value
    for part in path.parts:
        if part in DATASETS:
            return part
    return "unknown"


def variant_for(path):
    for variant in ("raw", "normalized", "joint"):
        if variant in path.parts:
            return variant
    return "joint"


def read_root(root, pipeline):
    rows = []
    if not root.exists():
        return rows
    if pipeline == "hybrid":
        time_key = "hybridModelInferenceTimeNs"
        calls_key = "hybridModelInferenceCalls"
    else:
        time_key = "msIndexPPModelInferenceTimeNs"
        calls_key = "msIndexPPModelInferenceCalls"

    for path in root.rglob("runs.csv"):
        if pipeline == "approximate" and "timing_runs" not in path.parts:
            # Evaluation runs execute the exact and approximate traversals
            # together to compute recall; use the independent timing runs for
            # the denominator of the overhead percentage.
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        if not csv_rows:
            continue
        row = csv_rows[-1]
        calls = as_float(row, calls_key)
        inference_ns = as_float(row, time_key)
        if calls <= 0 or inference_ns < 0:
            continue
        query_time_ms = as_float(row, "queryTime")
        if pipeline == "hybrid":
            evaluation_csv = path.parent / "evaluation.csv"
            if evaluation_csv.exists():
                with evaluation_csv.open("r", encoding="utf-8", newline="") as evaluation_handle:
                    evaluation_rows = list(csv.DictReader(evaluation_handle))
                query_time_ms = sum(as_float(item, "hybrid_total_time_ms") for item in evaluation_rows)
        rows.append(
            {
                "pipeline": pipeline,
                "dataset": dataset_for(row, path),
                "model_variant": variant_for(path),
                "representation": "normalized" if row.get("normalize", "").lower() == "true" else "raw",
                "k": int(as_float(row, "K")),
                "noise": as_float(row, "queryNoiseEps"),
                "query_time_ms": query_time_ms,
                "inference_time_ms": inference_ns / 1_000_000.0,
                "inference_calls": int(calls),
                "source": str(path),
            }
        )
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = read_root(args.hybrid_root, "hybrid") + read_root(args.approx_root, "approximate")
    run_fields = [
        "pipeline", "dataset", "model_variant", "representation", "k", "noise", "query_time_ms",
        "inference_time_ms", "inference_calls", "source",
    ]
    write_csv(args.output_dir / "inference_overhead_runs.csv", rows, run_fields)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["pipeline"], row["dataset"], row["model_variant"], row["representation"], row["k"])].append(row)
    summary = []
    for (pipeline, dataset, model_variant, representation, k), group in sorted(groups.items()):
        query_time = sum(row["query_time_ms"] for row in group)
        inference_time = sum(row["inference_time_ms"] for row in group)
        calls = sum(row["inference_calls"] for row in group)
        summary.append(
            {
                "pipeline": pipeline,
                "dataset": dataset,
                "model_variant": model_variant,
                "representation": representation,
                "k": k,
                "runs": len(group),
                "query_time_ms": round(query_time, 6),
                "inference_time_ms": round(inference_time, 6),
                "inference_calls": calls,
                "inference_pct_of_query": round(100.0 * inference_time / query_time, 6)
                if query_time > 0 else "",
                "avg_inference_us_per_call": round(1000.0 * inference_time / calls, 6)
                if calls > 0 else "",
            }
        )
    write_csv(
        args.output_dir / "inference_overhead_summary.csv",
        summary,
        [
            "pipeline", "dataset", "model_variant", "representation", "k", "runs", "query_time_ms",
            "inference_time_ms", "inference_calls", "inference_pct_of_query",
            "avg_inference_us_per_call",
        ],
    )
    print(f"Loaded {len(rows)} runs; wrote {len(summary)} grouped rows")


if __name__ == "__main__":
    main()
