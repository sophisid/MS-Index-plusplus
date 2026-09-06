#!/usr/bin/env python3.10
"""Plot requested recall against achieved AMS-Index recall."""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_DATASETS = ("synthetic", "stocks", "wind", "weather", "DuckDuckGeese")
REPRESENTATIONS = ("raw", "normalized")
KS = (1, 10, 100)
K_STYLES = {
    1: ("#007C91", "o"),
    10: ("#E68632", "s"),
    100: ("#6A7F3F", "D"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("artifacts_local_behavior/ams"),
        help="AMS results root. Both local and official variants layouts are detected.",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_local_behavior/recall_target_vs_actual"),
    )
    parser.add_argument("--basename", default="target_vs_actual_recall")
    return parser.parse_args()


def dataset_label(dataset):
    return {
        "synthetic": "Synthetic",
        "stocks": "Stocks",
        "wind": "Wind",
        "weather": "Weather",
        "DuckDuckGeese": "DuckDuckGeese",
    }.get(dataset, dataset)


def load_jsonl(path):
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON in {}:{}: {}".format(path, line_number, error))
    return rows


def candidate_files(results_root, dataset):
    dataset_root = results_root / dataset
    return (
        (3, dataset_root / dataset / "evaluation_results.jsonl"),
        (3, dataset_root / "evaluation_results.jsonl"),
        (2, dataset_root / "variants" / "joint" / "evaluation_results.jsonl"),
        (1, dataset_root / "variants" / "raw" / "evaluation_results.jsonl"),
        (1, dataset_root / "variants" / "normalized" / "evaluation_results.jsonl"),
    )


def load_results(results_root, datasets):
    # Higher-priority layouts replace representation-specific fallback files.
    merged = {}
    for dataset in datasets:
        for priority, path in candidate_files(results_root, dataset):
            for row in load_jsonl(path):
                if not bool(row.get("from_index", True)):
                    continue
                representation = "normalized" if bool(row.get("normalize")) else "raw"
                key = (
                    dataset,
                    representation,
                    int(row["k"]),
                    round(float(row["noise"]), 4),
                )
                if key not in merged or priority >= merged[key][0]:
                    merged[key] = (priority, row, path)
    return {key: value[1] for key, value in merged.items()}


def summary_rows(results, target_recall):
    rows = []
    for (dataset, representation, k, noise), result in sorted(results.items()):
        actual = float(result["avg_accuracy"])
        rows.append(
            {
                "dataset": dataset,
                "representation": representation,
                "k": k,
                "noise": noise,
                "target_recall": round(target_recall, 6),
                "actual_recall": round(actual, 6),
                "recall_gap": round(actual - target_recall, 6),
                "meets_target": actual >= target_recall,
                "early_stop_rate": result.get("early_stop_rate", ""),
                "query_time_ms": result.get("query_time_ms", ""),
            }
        )
    return rows


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "representation",
        "k",
        "noise",
        "target_recall",
        "actual_recall",
        "recall_gap",
        "meets_target",
        "early_stop_rate",
        "query_time_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, datasets, target_recall, pdf_path, png_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise SystemExit("matplotlib is required: {}".format(error))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )

    actual_values = [float(row["actual_recall"]) for row in rows]
    lower_value = min(actual_values + [target_recall]) if actual_values else target_recall
    upper_value = max(actual_values + [target_recall]) if actual_values else target_recall
    y_min = max(0.0, lower_value - 0.06)
    y_max = min(1.02, max(1.0, upper_value + 0.03))

    figure, axes = plt.subplots(
        len(datasets),
        len(REPRESENTATIONS),
        figsize=(9.2, 2.25 * len(datasets) + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    by_panel = {}
    for row in rows:
        by_panel.setdefault((row["dataset"], row["representation"]), []).append(row)

    for dataset_index, dataset in enumerate(datasets):
        for representation_index, representation in enumerate(REPRESENTATIONS):
            axis = axes[dataset_index][representation_index]
            axis.axhspan(y_min, target_recall, color="#C84B31", alpha=0.055, linewidth=0)
            axis.axhline(target_recall, color="#303030", linestyle="--", linewidth=1.2, zorder=2)
            panel_rows = by_panel.get((dataset, representation), [])
            for k in KS:
                selected = sorted(
                    (row for row in panel_rows if int(row["k"]) == k),
                    key=lambda row: float(row["noise"]),
                )
                if not selected:
                    continue
                color, marker = K_STYLES[k]
                axis.plot(
                    [float(row["noise"]) for row in selected],
                    [float(row["actual_recall"]) for row in selected],
                    color=color,
                    marker=marker,
                    markersize=4.2,
                    linewidth=1.35,
                    label="$k={}$".format(k),
                    zorder=3,
                )
                missed = [row for row in selected if float(row["actual_recall"]) < target_recall]
                if missed:
                    axis.scatter(
                        [float(row["noise"]) for row in missed],
                        [float(row["actual_recall"]) for row in missed],
                        facecolors="none",
                        edgecolors="#B3261E",
                        s=38,
                        linewidths=1.0,
                        zorder=4,
                    )

            if not panel_rows:
                axis.text(
                    0.5,
                    0.5,
                    "No data yet",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                )
            axis.set_xlim(-0.02, 0.92)
            axis.set_ylim(y_min, y_max)
            axis.set_xticks([value / 10 for value in range(10)])
            axis.grid(axis="y", color="#D8D2C4", linewidth=0.55, alpha=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if dataset_index == 0:
                axis.set_title(representation.capitalize(), fontweight="bold")
            if representation_index == 0:
                axis.set_ylabel("{}\nActual recall@$k$".format(dataset_label(dataset)))
            if dataset_index == len(datasets) - 1:
                axis.set_xlabel("Query noise")

    legend_handles = [
        Line2D([0], [0], color=K_STYLES[k][0], marker=K_STYLES[k][1], linewidth=1.35, label="$k={}$".format(k))
        for k in KS
    ]
    legend_handles.append(
        Line2D([0], [0], color="#303030", linestyle="--", linewidth=1.2, label="Target recall = {:.2f}".format(target_recall))
    )
    figure.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    figure.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.07, hspace=0.24, wspace=0.10)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if not 0.0 <= args.target_recall <= 1.0:
        raise SystemExit("--target-recall must be between 0 and 1")
    results = load_results(args.results_root, args.datasets)
    rows = summary_rows(results, args.target_recall)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / (args.basename + ".csv")
    pdf_path = args.output_dir / (args.basename + ".pdf")
    png_path = args.output_dir / (args.basename + ".png")
    write_summary(csv_path, rows)
    plot(rows, args.datasets, args.target_recall, pdf_path, png_path)
    print(csv_path)
    print(pdf_path)
    print(png_path)
    print("Loaded {} evaluation points".format(len(rows)))


if __name__ == "__main__":
    main()
