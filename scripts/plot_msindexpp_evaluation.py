#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def overall_accuracy(rows):
    correct = sum(int(row["prediction_correct"]) for row in rows)
    return correct / len(rows), correct


def plot_accuracy(rows, output_dir: Path):
    accuracy, correct = overall_accuracy(rows)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Correct", "Incorrect"], [correct, len(rows) - correct], color=["#2a9d8f", "#e76f51"])
    ax.set_title(f"Decision Accuracy: {accuracy:.3f}")
    ax.set_ylabel("Queries")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(rows, output_dir: Path):
    labels = ["MASS", "MSINDEX"]
    counts = Counter((row["actual_best_algorithm"], row["predicted_algorithm"]) for row in rows)
    matrix = [
        [counts[("MASS", "MASS")], counts[("MASS", "MSINDEX")]],
        [counts[("MSINDEX", "MASS")], counts[("MSINDEX", "MSINDEX")]],
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual Best")
    ax.set_title("Decision Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def plot_runtime_scatter(rows, output_dir: Path):
    mass_times = [float(row["mass_time_ms"]) for row in rows]
    msindex_times = [float(row["msindex_time_ms"]) for row in rows]
    colors = ["#2a9d8f" if int(row["prediction_correct"]) else "#e76f51" for row in rows]

    max_time = max(max(mass_times), max(msindex_times))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mass_times, msindex_times, c=colors, alpha=0.75, edgecolors="none")
    ax.plot([0, max_time], [0, max_time], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("MASS time (ms)")
    ax.set_ylabel("MSINDEX time (ms)")
    ax.set_title("Per-query Runtime Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_scatter.png", dpi=200)
    plt.close(fig)


def plot_margin_histogram(rows, output_dir: Path):
    margins = [abs(float(row["mass_time_ms"]) - float(row["msindex_time_ms"])) for row in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(margins, bins=20, color="#457b9d", edgecolor="white")
    ax.set_xlabel("Absolute runtime gap (ms)")
    ax.set_ylabel("Queries")
    ax.set_title("How Different Were MASS and MSINDEX?")
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_gap_histogram.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot MSIndexPP decision evaluation results.")
    parser.add_argument("--input", required=True, help="Evaluation CSV produced by Java.")
    parser.add_argument("--output-dir", required=True, help="Directory where plots will be written.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)

    plot_accuracy(rows, output_dir)
    plot_confusion_matrix(rows, output_dir)
    plot_runtime_scatter(rows, output_dir)
    plot_margin_histogram(rows, output_dir)

    accuracy, correct = overall_accuracy(rows)
    print(f"rows={len(rows)}")
    print(f"correct={correct}")
    print(f"accuracy={accuracy:.4f}")
    print(f"plots={output_dir}")


if __name__ == "__main__":
    main()
