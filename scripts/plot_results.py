#!/usr/bin/env python3
"""
plot_results.py

Generates evaluation plots matching the original suite format.

Output (in --output-dir):
  query_time_k{1,10,100}.pdf  — avg query time vs noise per algorithm,
                                 grid: 2 rows (non-norm / norm) x datasets
  accuracy_k{1,10,100}.pdf   — MS-Index++ avg accuracy vs noise,
                                 grid: 2 rows x datasets
  training_time.pdf           — training time per dataset (stacked bar)

Usage:
  python3 scripts/plot_results.py [--artifact-root DIR] [--output-dir DIR]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── palette / labels ──────────────────────────────────────────────────────────
COLOR = {"MSINDEX": "#4C72B0", "MASS": "#DD8452", "MSINDEXPP": "#55A868"}
LABEL = {"MSINDEX": "MS-Index", "MASS": "MASS", "MSINDEXPP": "MS-Index++"}
MARKER = {"MSINDEX": "o", "MASS": "s", "MSINDEXPP": "^"}
ALGOS = ["MSINDEX", "MASS", "MSINDEXPP"]
K_VALUES = [1, 10, 100]
NOISE_VALUES = [round(i * 0.1, 1) for i in range(10)]


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def save(fig, path):
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("saved %s" % path)


# ── data collection ───────────────────────────────────────────────────────────
def avg(lst):
    return sum(lst) / len(lst) if lst else None


def collect_timing(artifact_root):
    """
    Returns:
      timing[dataset][k][normalize][algo][noise] = avg query_time_ms
                                                   (averaged over from_index)
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)
                )
            )
        )
    )
    for d in sorted(artifact_root.iterdir()):
        if not d.is_dir():
            continue
        for row in load_jsonl(d / "timing_results.jsonl"):
            algo  = row.get("algorithm")
            k     = row.get("k")
            norm  = row.get("normalize")
            noise = round(float(row.get("noise", 0)), 1)
            qt    = row.get("query_time_ms")
            if algo and k is not None and norm is not None and qt is not None:
                data[d.name][int(k)][bool(norm)][algo][noise].append(float(qt))
    # collapse to avg
    result = {}
    for ds, kd in data.items():
        result[ds] = {}
        for k, nd in kd.items():
            result[ds][k] = {}
            for norm, ad in nd.items():
                result[ds][k][norm] = {}
                for algo, noised in ad.items():
                    result[ds][k][norm][algo] = {
                        noise: avg(vals) for noise, vals in noised.items()
                    }
    return result


def collect_evaluation(artifact_root):
    """
    Returns:
      evaluation[dataset][k][normalize][noise] = avg accuracy
                                                 (averaged over from_index)
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )
    for d in sorted(artifact_root.iterdir()):
        if not d.is_dir():
            continue
        for row in load_jsonl(d / "evaluation_results.jsonl"):
            k     = row.get("k")
            norm  = row.get("normalize")
            noise = round(float(row.get("noise", 0)), 1)
            acc   = row.get("avg_accuracy")
            if k is not None and norm is not None and acc is not None:
                data[d.name][int(k)][bool(norm)][noise].append(float(acc))
    result = {}
    for ds, kd in data.items():
        result[ds] = {}
        for k, nd in kd.items():
            result[ds][k] = {}
            for norm, noised in nd.items():
                result[ds][k][norm] = {
                    noise: avg(vals) for noise, vals in noised.items()
                }
    return result


def collect_training(artifact_root):
    training = {}
    for d in sorted(artifact_root.iterdir()):
        rt = d / "training_runtime.json"
        if d.is_dir() and rt.exists():
            info = json.loads(rt.read_text(encoding="utf-8"))
            training[d.name] = {
                "collection": info.get("collection_wall_time_s", 0.0),
                "fit":        info.get("trainer_wall_time_s",    0.0),
            }
    return training


# ── subplot grid helpers ──────────────────────────────────────────────────────
def make_grid(datasets, n_rows=2, subplot_w=3.0, subplot_h=2.8):
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(subplot_w * n_cols, subplot_h * n_rows),
        squeeze=False,
    )
    return fig, axes


def style_subplot(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=7, pad=3)
    ax.set_xlabel("noise", fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(labelsize=5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, linewidth=0.4, alpha=0.5)


# ── plot 1: query time ────────────────────────────────────────────────────────
def plot_query_time(timing, out_dir):
    for k in K_VALUES:
        datasets = sorted(ds for ds in timing if k in timing[ds])
        if not datasets:
            continue

        fig, axes = make_grid(datasets)
        row_labels = {0: "non-normalized", 1: "normalized"}

        for col, ds in enumerate(datasets):
            for row_i, norm in enumerate([False, True]):
                ax = axes[row_i][col]
                algo_data = timing[ds].get(k, {}).get(norm, {})
                has_data = False
                for algo in ALGOS:
                    noise_map = algo_data.get(algo, {})
                    xs = [n for n in NOISE_VALUES if n in noise_map]
                    ys = [noise_map[n] for n in xs]
                    if xs:
                        ax.plot(xs, ys, marker=MARKER[algo], markersize=3,
                                linewidth=1.0, color=COLOR[algo], label=LABEL[algo])
                        has_data = True

                ylabel = "ms" if col == 0 else ""
                style_subplot(ax, ds if row_i == 0 else "", ylabel)
                if not has_data:
                    ax.set_visible(False)

            # row label on leftmost column
            if col == 0:
                for row_i, lbl in row_labels.items():
                    axes[row_i][0].set_ylabel(lbl + "\nms", fontsize=6)

        # shared legend
        handles = [
            plt.Line2D([0], [0], color=COLOR[a], marker=MARKER[a],
                       markersize=4, linewidth=1, label=LABEL[a])
            for a in ALGOS
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(ALGOS), fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Avg query time  |  k=%d" % k, fontsize=9, y=1.01)
        fig.tight_layout()
        save(fig, out_dir / ("query_time_k%d.pdf" % k))


# ── plot 2: accuracy ──────────────────────────────────────────────────────────
def plot_accuracy(evaluation, out_dir):
    for k in K_VALUES:
        datasets = sorted(ds for ds in evaluation if k in evaluation[ds])
        if not datasets:
            continue

        fig, axes = make_grid(datasets)

        for col, ds in enumerate(datasets):
            for row_i, norm in enumerate([False, True]):
                ax = axes[row_i][col]
                noise_map = evaluation[ds].get(k, {}).get(norm, {})
                xs = [n for n in NOISE_VALUES if n in noise_map]
                ys = [noise_map[n] for n in xs]
                if xs:
                    ax.plot(xs, ys, marker="^", markersize=3, linewidth=1.0,
                            color=COLOR["MSINDEXPP"], label=LABEL["MSINDEXPP"])
                    ax.axhline(0.9, color="red", linewidth=0.7,
                               linestyle="--", alpha=0.8)

                ylabel = "accuracy" if col == 0 else ""
                style_subplot(ax, ds if row_i == 0 else "", ylabel, ylim=(0, 1.05))
                if not xs:
                    ax.set_visible(False)

            if col == 0:
                axes[0][0].set_ylabel("non-normalized\naccuracy", fontsize=6)
                axes[1][0].set_ylabel("normalized\naccuracy", fontsize=6)

        handles = [
            plt.Line2D([0], [0], color=COLOR["MSINDEXPP"], marker="^",
                       markersize=4, linewidth=1, label=LABEL["MSINDEXPP"]),
            plt.Line2D([0], [0], color="red", linewidth=0.8,
                       linestyle="--", label="target 0.9"),
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("MS-Index++ accuracy  |  k=%d" % k, fontsize=9, y=1.01)
        fig.tight_layout()
        save(fig, out_dir / ("accuracy_k%d.pdf" % k))


# ── plot 3: training time ─────────────────────────────────────────────────────
def plot_training_time(training, out_dir):
    datasets = sorted(training)
    if not datasets:
        return
    collection_min = [training[d]["collection"] / 60.0 for d in datasets]
    fit_min        = [training[d]["fit"]        / 60.0 for d in datasets]
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.6), 5))
    ax.bar(x, collection_min, label="Data collection", color=COLOR["MSINDEX"])
    ax.bar(x, fit_min, bottom=collection_min, label="GBDT fit", color=COLOR["MSINDEXPP"])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Time (minutes)")
    ax.set_title("MS-Index++ training time per dataset")
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir / "training_time.pdf")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts" / "msindexpp_suite")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts" / "msindexpp_suite" / "plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timing     = collect_timing(args.artifact_root)
    evaluation = collect_evaluation(args.artifact_root)
    training   = collect_training(args.artifact_root)

    if not timing and not evaluation and not training:
        print("No results found under %s" % args.artifact_root, file=sys.stderr)
        sys.exit(1)

    if timing:
        plot_query_time(timing, args.output_dir)
    if evaluation:
        plot_accuracy(evaluation, args.output_dir)
    if training:
        plot_training_time(training, args.output_dir)

    print("Done. Plots in %s" % args.output_dir)


if __name__ == "__main__":
    main()
