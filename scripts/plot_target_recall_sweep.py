#!/usr/bin/env python3.10
"""Create achieved-recall and speedup bar charts for an AMS target sweep."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASETS = ("synthetic", "stocks", "wind", "weather", "DuckDuckGeese")
DEFAULT_TARGETS = (0.80, 0.85, 0.90, 0.95, 0.99)
# Figure 11 / the requested regressor sweep reports k=10 and k=100.
KS = (10, 100)
K_COLORS = {1: "#147D92", 10: "#E68632", 100: "#718B3A"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("artifacts_target_specific"),
    )
    parser.add_argument("--targets", nargs="+", type=float, default=list(DEFAULT_TARGETS))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=("raw", "normalized"),
        default=["raw", "normalized"],
    )
    parser.add_argument(
        "--noise-values",
        nargs="+",
        type=float,
        default=None,
        help="Only aggregate selected query-noise values. Default: all available values.",
    )
    parser.add_argument(
        "--speedup-baseline",
        choices=("fastest", "mass", "msindex"),
        default="fastest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_target_specific/plots"),
    )
    return parser.parse_args()


def dataset_label(dataset):
    return {
        "synthetic": "Synthetic",
        "stocks": "Stocks",
        "wind": "Wind",
        "weather": "Weather",
        "DuckDuckGeese": "DuckDuckGeese",
    }.get(dataset, dataset)


def target_directory(results_root, target):
    tag = ("{:.2f}".format(target)).replace(".", "p")
    return results_root / ("target_" + tag)


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
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


def result_directory(results_root, target, dataset):
    base = target_directory(results_root, target) / dataset
    nested = base / dataset
    return nested if nested.is_dir() else base


def selected_noise(noise, requested_noises):
    if requested_noises is None:
        return True
    return any(abs(noise - requested) < 1e-8 for requested in requested_noises)


def speedup_value(timings, baseline):
    ams = timings.get("MSINDEXPP")
    mass = timings.get("MASS")
    msindex = timings.get("MSINDEX")
    if ams in (None, 0):
        return None
    if baseline == "mass":
        reference = mass
    elif baseline == "msindex":
        reference = msindex
    else:
        available = [value for value in (mass, msindex) if value is not None]
        reference = min(available) if available else None
    return reference / ams if reference is not None else None


def load_points(results_root, targets, datasets, requested_noises, baseline):
    points = []
    for target in targets:
        for dataset in datasets:
            root = result_directory(results_root, target, dataset)
            evaluation_rows = read_jsonl(root / "evaluation_results.jsonl")
            timing_rows = read_jsonl(root / "timing_results.jsonl")
            timings = defaultdict(dict)
            for row in timing_rows:
                if not bool(row.get("from_index", True)):
                    continue
                noise = round(float(row["noise"]), 4)
                if not selected_noise(noise, requested_noises):
                    continue
                representation = "normalized" if bool(row.get("normalize")) else "raw"
                key = (representation, int(row["k"]), noise)
                timings[key][row["algorithm"]] = float(row["query_time_ms"])

            for row in evaluation_rows:
                if not bool(row.get("from_index", True)):
                    continue
                noise = round(float(row["noise"]), 4)
                if not selected_noise(noise, requested_noises):
                    continue
                representation = "normalized" if bool(row.get("normalize")) else "raw"
                k = int(row["k"])
                if k not in KS:
                    continue
                timing = timings.get((representation, k, noise), {})
                speedup = speedup_value(timing, baseline)
                points.append(
                    {
                        "target_recall": target,
                        "dataset": dataset,
                        "representation": representation,
                        "k": k,
                        "noise": noise,
                        "actual_recall": float(row["avg_accuracy"]),
                        "recall_gap": float(row["avg_accuracy"]) - target,
                        "early_stop_rate": float(row.get("early_stop_rate", 0.0)),
                        "mass_time_ms": timing.get("MASS"),
                        "msindex_time_ms": timing.get("MSINDEX"),
                        "ams_time_ms": timing.get("MSINDEXPP"),
                        "speedup": speedup,
                    }
                )
    return points


def aggregate_points(points):
    grouped = defaultdict(list)
    for point in points:
        key = (
            point["target_recall"],
            point["dataset"],
            point["representation"],
            point["k"],
        )
        grouped[key].append(point)

    aggregated = []
    for (target, dataset, representation, k), group in sorted(grouped.items()):
        speedups = [point["speedup"] for point in group if point["speedup"] is not None]
        aggregated.append(
            {
                "target_recall": target,
                "dataset": dataset,
                "representation": representation,
                "k": k,
                "noise_points": len(group),
                "actual_recall_mean": statistics.mean(point["actual_recall"] for point in group),
                "actual_recall_min": min(point["actual_recall"] for point in group),
                "actual_recall_max": max(point["actual_recall"] for point in group),
                "recall_gap_mean": statistics.mean(point["recall_gap"] for point in group),
                "early_stop_rate_mean": statistics.mean(point["early_stop_rate"] for point in group),
                "speedup_mean": statistics.mean(speedups) if speedups else None,
                "speedup_min": min(speedups) if speedups else None,
                "speedup_max": max(speedups) if speedups else None,
            }
        )
    return aggregated


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def plot_dataset(dataset, rows, targets, representations, baseline, output_dir, pdf_pages):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(
        len(representations),
        2,
        figsize=(9.0, 3.2 * len(representations) + 0.8),
        squeeze=False,
    )
    x = np.arange(len(targets), dtype=float)
    width = 0.23
    offsets = {k: (index - 1) * width for index, k in enumerate(KS)}
    dataset_rows = [row for row in rows if row["dataset"] == dataset]

    for row_index, representation in enumerate(representations):
        recall_axis = axes[row_index][0]
        speed_axis = axes[row_index][1]
        representation_rows = [row for row in dataset_rows if row["representation"] == representation]

        for k in KS:
            k_rows = {(row["target_recall"]): row for row in representation_rows if row["k"] == k}
            recall_values = [
                k_rows[target]["actual_recall_mean"] if target in k_rows else float("nan")
                for target in targets
            ]
            speedup_values = [
                k_rows[target]["speedup_mean"]
                if target in k_rows and k_rows[target]["speedup_mean"] is not None
                else float("nan")
                for target in targets
            ]
            recall_axis.bar(
                x + offsets[k],
                recall_values,
                width,
                color=K_COLORS[k],
                edgecolor="#202020",
                linewidth=0.35,
                label="$k={}$".format(k),
            )
            speed_axis.bar(
                x + offsets[k],
                speedup_values,
                width,
                color=K_COLORS[k],
                edgecolor="#202020",
                linewidth=0.35,
            )

        recall_axis.plot(
            x,
            targets,
            color="#262626",
            linestyle="--",
            marker="_",
            markersize=11,
            linewidth=1.2,
            label="Requested recall",
            zorder=4,
        )
        speed_axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)

        recall_values = [
            row["actual_recall_mean"] for row in representation_rows if row["actual_recall_mean"] is not None
        ]
        recall_floor = min(recall_values + list(targets)) if recall_values else min(targets)
        recall_axis.set_ylim(max(0.0, recall_floor - 0.06), 1.015)
        speed_values = [
            row["speedup_mean"] for row in representation_rows if row["speedup_mean"] is not None
        ]
        speed_axis.set_ylim(0.0, max(1.15, max(speed_values, default=1.0) * 1.18))

        recall_axis.set_title("{}: Achieved Recall".format(representation.capitalize()), fontweight="bold")
        speed_axis.set_title("{}: Speedup".format(representation.capitalize()), fontweight="bold")
        recall_axis.set_ylabel("Actual recall@$k$")
        speed_axis.set_ylabel("Times faster")
        for axis in (recall_axis, speed_axis):
            axis.set_xticks(x)
            axis.set_xticklabels(["{:.2f}".format(target) for target in targets])
            axis.set_xlabel("Target recall")
            axis.grid(axis="y", linestyle=":", linewidth=0.6, color="#AAA49A")
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        if not representation_rows:
            for axis in (recall_axis, speed_axis):
                axis.text(0.5, 0.5, "No data yet", transform=axis.transAxes, ha="center", va="center")

    legend = [Patch(facecolor=K_COLORS[k], edgecolor="#202020", label="$k={}$".format(k)) for k in KS]
    figure.legend(handles=legend, loc="upper center", ncol=len(KS), frameon=False, bbox_to_anchor=(0.5, 0.995))
    figure.suptitle(
        "{}: recall target and AMS-Index speedup vs {}".format(dataset_label(dataset), baseline),
        y=0.945,
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.10, hspace=0.48, wspace=0.25)

    safe_name = dataset.lower().replace("-", "_")
    pdf_path = output_dir / (safe_name + "_target_recall_sweep.pdf")
    png_path = output_dir / (safe_name + "_target_recall_sweep.png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    pdf_pages.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main():
    args = parse_args()
    targets = sorted(set(round(target, 4) for target in args.targets))
    if any(target <= 0.0 or target > 1.0 for target in targets):
        raise SystemExit("Targets must be in (0, 1]")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as error:
        raise SystemExit("matplotlib is required: {}".format(error))

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    points = load_points(
        args.results_root,
        targets,
        args.datasets,
        args.noise_values,
        args.speedup_baseline,
    )
    aggregated = aggregate_points(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point_fields = [
        "target_recall",
        "dataset",
        "representation",
        "k",
        "noise",
        "actual_recall",
        "recall_gap",
        "early_stop_rate",
        "mass_time_ms",
        "msindex_time_ms",
        "ams_time_ms",
        "speedup",
    ]
    aggregate_fields = [
        "target_recall",
        "dataset",
        "representation",
        "k",
        "noise_points",
        "actual_recall_mean",
        "actual_recall_min",
        "actual_recall_max",
        "recall_gap_mean",
        "early_stop_rate_mean",
        "speedup_mean",
        "speedup_min",
        "speedup_max",
    ]
    points_csv = args.output_dir / "target_recall_sweep_points.csv"
    aggregate_csv = args.output_dir / "target_recall_sweep_summary.csv"
    write_csv(points_csv, points, point_fields)
    write_csv(aggregate_csv, aggregated, aggregate_fields)

    all_pdf = args.output_dir / "target_recall_sweep_all_datasets.pdf"
    generated = []
    with PdfPages(all_pdf) as pdf_pages:
        for dataset in args.datasets:
            generated.extend(
                plot_dataset(
                    dataset,
                    aggregated,
                    targets,
                    args.representations,
                    args.speedup_baseline,
                    args.output_dir,
                    pdf_pages,
                )
            )

    print(points_csv)
    print(aggregate_csv)
    print(all_pdf)
    for path in generated:
        print(path)
    print("Loaded {} target/noise evaluation points".format(len(points)))


if __name__ == "__main__":
    main()
