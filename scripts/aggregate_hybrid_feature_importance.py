#!/usr/bin/env python3
"""Aggregate Hybrid feature importance values across datasets and variants."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("synthetic", "stocks", "wind", "weather", "DuckDuckGeese")
VARIANTS = ("raw", "normalized", "joint")


def read_importance(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["feature"]: float(row["importance_percent"])
            for row in csv.DictReader(handle)
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = defaultdict(list)
    missing = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            path = args.results_root / dataset / "models" / variant / "feature_importance.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            for feature, importance in read_importance(path).items():
                values[(variant, feature)].append((dataset, importance))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "feature", "mean_importance_percent", "std_importance_percent", "datasets"])
        for variant in VARIANTS:
            features = sorted({feature for current_variant, feature in values if current_variant == variant})
            for feature in features:
                entries = values[(variant, feature)]
                importances = [importance for _, importance in entries]
                std = statistics.stdev(importances) if len(importances) > 1 else 0.0
                writer.writerow([
                    variant,
                    feature,
                    "{:.6f}".format(statistics.mean(importances)),
                    "{:.6f}".format(std),
                    len(entries),
                ])

    print("Wrote {}".format(args.output))
    if missing:
        print("Missing importance files:")
        for path in missing:
            print("  {}".format(path))


if __name__ == "__main__":
    main()
