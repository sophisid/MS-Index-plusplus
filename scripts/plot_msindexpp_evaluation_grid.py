#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_run_metadata(path: Path):
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 7:
        raise ValueError(f"Unexpected evaluation filename format: {path.name}")

    return {
        "k": int(parts[0][1:]),
        "normalize": parts[2] == "true",
        "seed": int(parts[3][4:]),
        "noise": float(parts[4][5:]),
        "query_from_index": parts[6] == "true",
    }


def load_rows(input_dir: Path):
    grouped = defaultdict(list)

    for path in sorted(input_dir.glob("*.csv")):
        metadata = parse_run_metadata(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            combined = dict(row)
            combined.update(metadata)
            grouped[(metadata["k"], metadata["normalize"])].append(combined)

    if not grouped:
        raise ValueError(f"No evaluation CSV files found in {input_dir}")
    return grouped


def aggregate_by_noise(rows):
    by_noise = defaultdict(list)
    for row in rows:
        by_noise[row["noise"]].append(row)

    noise_values = sorted(by_noise)
    accuracy = []
    regret = []

    for noise in noise_values:
        subset = by_noise[noise]
        correct = sum(int(row["prediction_correct"]) for row in subset)
        accuracy.append(correct / len(subset))

        regrets = []
        for row in subset:
            predicted = float(row["predicted_time_ms"])
            best = min(float(row["mass_time_ms"]), float(row["msindex_time_ms"]))
            regrets.append(predicted - best)
        regret.append(sum(regrets) / len(regrets))

    return noise_values, accuracy, regret


def aggregate_by_source(rows):
    sources = [True, False]
    labels = ["from index", "not from index"]
    accuracy = []
    regret = []

    for source in sources:
        subset = [row for row in rows if row["query_from_index"] == source]
        if not subset:
            accuracy.append(0)
            regret.append(0)
            continue

        correct = sum(int(row["prediction_correct"]) for row in subset)
        accuracy.append(correct / len(subset))

        regrets = []
        for row in subset:
            predicted = float(row["predicted_time_ms"])
            best = min(float(row["mass_time_ms"]), float(row["msindex_time_ms"]))
            regrets.append(predicted - best)
        regret.append(sum(regrets) / len(regrets))

    return labels, accuracy, regret


def plot_group(rows, k, normalize, output_path: Path):
    noise_values, noise_accuracy, noise_regret = aggregate_by_noise(rows)
    source_labels, source_accuracy, source_regret = aggregate_by_source(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"MSIndexPP Routing Evaluation | K={k} | normalize={str(normalize).lower()}")

    ax = axes[0, 0]
    ax.bar([str(n) for n in noise_values], noise_accuracy, color="#2a9d8f")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy by Noise")
    ax.set_xlabel("queryNoiseEps")
    ax.set_ylabel("Accuracy")

    ax = axes[0, 1]
    ax.bar([str(n) for n in noise_values], noise_regret, color="#457b9d")
    ax.set_title("Avg Extra Time vs Best Choice by Noise")
    ax.set_xlabel("queryNoiseEps")
    ax.set_ylabel("Predicted time - best time (ms)")

    ax = axes[1, 0]
    ax.bar(source_labels, source_accuracy, color=["#e9c46a", "#f4a261"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy by Query Source")
    ax.set_ylabel("Accuracy")

    ax = axes[1, 1]
    ax.bar(source_labels, source_regret, color=["#8ecae6", "#219ebc"])
    ax.set_title("Avg Extra Time vs Best Choice by Query Source")
    ax.set_ylabel("Predicted time - best time (ms)")

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated MSIndexPP evaluation results.")
    parser.add_argument("--input-dir", required=True, help="Directory containing evaluation CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory where the six summary images will be written.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_rows(input_dir)

    for (k, normalize), rows in sorted(grouped.items()):
        output_path = output_dir / f"k{k}_norm_{str(normalize).lower()}.png"
        plot_group(rows, k, normalize, output_path)
        print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
