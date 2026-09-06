#!/usr/bin/env python3.10
"""Summarize and plot TIA top-1 distance estimates versus exact distances."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DATASETS = ("synthetic", "stocks", "wind", "weather", "DuckDuckGeese")
TARGETS = (0.80, 0.85, 0.90, 0.95, 0.99)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("artifacts_target_independent"))
    parser.add_argument("--targets", nargs="+", type=float, default=list(TARGETS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_target_independent/plots"))
    return parser.parse_args()


def close(left, right):
    return abs(float(left) - float(right)) < 1e-7


def load_rows(results_root, targets, datasets):
    wanted_targets = [round(target, 4) for target in targets]
    rows = []
    for path in results_root.rglob("evaluation_trace.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}")
                dataset = row.get("dataset")
                target = float(row.get("target_recall", -1.0))
                if dataset not in datasets or not any(close(target, value) for value in wanted_targets):
                    continue
                if int(row.get("requested_k", row.get("k", 0))) != 1:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "target_recall": round(target, 4),
                        "representation": "normalized" if bool(row.get("normalize")) else "raw",
                        "noise": float(row.get("query_noise_eps", row.get("noise", 0.0))),
                        "query_id": int(row.get("query_id", -1)),
                        "actual_recall": float(row["actual_recall"]),
                        "exact_top1_distance": float(row["exact_top_k_distance"]),
                        "returned_top1_distance": float(row["returned_top_k_distance"]),
                        "first_pass_top1_distance": float(row["first_pass_top_k_distance"]),
                    }
                )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "target_recall", "representation", "noise", "query_id",
        "actual_recall", "exact_top1_distance", "returned_top1_distance",
        "first_pass_top1_distance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path, rows, targets, datasets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#56B4E9", "#E69F00", "#CC79A7", "#718B3A", "#009E73"]
    figure, axes = plt.subplots(1, len(datasets), figsize=(3.1 * len(datasets), 3.0), squeeze=False)
    for index, dataset in enumerate(datasets):
        axis = axes[0][index]
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        maximum = max(
            [
                row[field]
                for row in dataset_rows
                for field in ("exact_top1_distance", "returned_top1_distance", "first_pass_top1_distance")
            ]
            or [1.0]
        )
        for target_index, target in enumerate(targets):
            selected = [row for row in dataset_rows if close(row["target_recall"], target)]
            if not selected:
                continue
            color = colors[target_index % len(colors)]
            for representation, marker in (("raw", "o"), ("normalized", "s")):
                points = [row for row in selected if row["representation"] == representation]
                if not points:
                    continue
                axis.scatter(
                    [row["returned_top1_distance"] for row in points],
                    [row["exact_top1_distance"] for row in points],
                    s=11,
                    alpha=0.42,
                    color=color,
                    marker=marker,
                    label=f"$R_t={target:.2f}$ {representation}" if index == 0 else None,
                )
                axis.scatter(
                    [row["first_pass_top1_distance"] for row in points],
                    [row["exact_top1_distance"] for row in points],
                    s=13,
                    alpha=0.30,
                    color=color,
                    marker="x",
                )
        axis.plot([0, maximum], [0, maximum], color="#222222", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xlabel("Returned / first-pass distance")
        if index == 0:
            axis.set_ylabel("Exact top-1 distance")
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    targets = sorted(set(round(target, 4) for target in args.targets))
    if any(target <= 0.0 or target > 1.0 for target in targets):
        raise SystemExit("Targets must be in (0, 1]")
    rows = load_rows(args.results_root, targets, args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "regressor_distance_points.csv", rows)
    if rows:
        make_plot(args.output_dir / "regressor_distance.pdf", rows, targets, args.datasets)
    print(f"Loaded {len(rows)} top-1 distance points")


if __name__ == "__main__":
    main()
