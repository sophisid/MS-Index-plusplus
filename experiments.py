#!/usr/bin/env python3
import itertools
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

# Αν θες μόνο plotting χωρίς να ξανατρέξεις τα πειράματα:
RUN_EXPERIMENTS = True
MAKE_PLOTS = True

# Όριο για να μη σκοτώσουμε τον υπολογιστή
MAX_RUNS = 800  # βάλε None για no-limit (με δική σου ευθύνη)

# Βάση από το bash script σου
BASE_PARAMS = {
    "N": 100,                    # number of time series
    "channels": -1,
    "nQueryChannels": -1,
    "qLen": 73,
    "maxM": 100000,
    "runtimeMode": "FULL_NO_STORE",
    "experimentId": 1,
    "seed": 100,
    "parallel": False,
    "queryFromIndexed": False,
}

# Πειραματικοί άξονες
ALGORITHMS = ["MSINDEX", "MSINDEXPP", "MASS"]
DATASETS = ["data/synthetic", "data/stocks", "data/weather"]
NORMALIZE_VALUES = [False, True]
NOISE_VALUES = [round(x, 1) for x in [0.1, 0.3, 0.5, 0.7, 0.9]]
K_VALUES = [1, 5, 10]
NQUERIES_VALUES = [10, 100, 500]
# Αν ποτέ θες παραπάνω:
# NQUERIES_VALUES = [10, 100, 1000, 10000, 100000, 1000000, 10000000]


# ============================================================
# Helpers
# ============================================================

def bool_to_java_str(b: bool) -> str:
    return "true" if b else "false"


def run_maven_package():
    """Τρέχει mvn clean package πριν από όλα τα experiments."""
    print("Running: mvn -q -DskipTests clean package")
    subprocess.run(
        ["mvn", "-q", "-DskipTests", "clean", "package"],
        check=True
    )


def run_single_experiment(algorithmType: str,
                          dataPath: str,
                          N: int,
                          maxM: int,
                          channels: int,
                          nQueryChannels: int,
                          qLen: int,
                          K: int,
                          normalize: bool,
                          nQueries: int,
                          runtimeMode: str,
                          experimentId: int,
                          seed: int,
                          parallel: bool,
                          queryFromIndexed: bool,
                          queryNoiseEps: float):
    """Τρέχει ΕΝΑ run του Java προγράμματος με τα δοσμένα arguments."""
    java_cmd = [
        "java",
        "-cp", "target/MS-Index-1.0-jar-with-dependencies.jar",
        "io.github.Main",
        algorithmType,
        dataPath,
        str(N),
        str(maxM),
        str(channels),
        str(nQueryChannels),
        str(qLen),
        str(K),
        bool_to_java_str(normalize),
        str(nQueries),
        runtimeMode,
        str(experimentId),
        str(seed),
        bool_to_java_str(parallel),
        bool_to_java_str(queryFromIndexed),
        str(queryNoiseEps),
    ]

    print("Running:", " ".join(java_cmd))
    subprocess.run(java_cmd, check=True)


def run_all_experiments():
    """Τρέχει όλα τα experiments σύμφωνα με τα grids παραπάνω."""

    run_maven_package()

    combinations = list(
        itertools.product(
            ALGORITHMS,
            DATASETS,
            NORMALIZE_VALUES,
            NOISE_VALUES,
            K_VALUES,
            NQUERIES_VALUES,
        )
    )

    print(f"Total potential runs: {len(combinations)}")

    runs_done = 0

    for (algorithmType,
         dataPath,
         normalize,
         noise,
         K,
         nQueries) in combinations:

        if MAX_RUNS is not None and runs_done >= MAX_RUNS:
            print(f"Reached MAX_RUNS={MAX_RUNS}, stopping further runs.")
            break

        print(
            f"\n=== Run #{runs_done + 1} ===\n"
            f"algorithmType={algorithmType}, dataPath={dataPath}, "
            f"normalize={normalize}, noise={noise}, K={K}, nQueries={nQueries}"
        )

        try:
            run_single_experiment(
                algorithmType=algorithmType,
                dataPath=dataPath,
                N=BASE_PARAMS["N"],
                maxM=BASE_PARAMS["maxM"],
                channels=BASE_PARAMS["channels"],
                nQueryChannels=BASE_PARAMS["nQueryChannels"],
                qLen=BASE_PARAMS["qLen"],
                K=K,
                normalize=normalize,
                nQueries=nQueries,
                runtimeMode=BASE_PARAMS["runtimeMode"],
                experimentId=BASE_PARAMS["experimentId"],
                seed=BASE_PARAMS["seed"],
                parallel=BASE_PARAMS["parallel"],
                queryFromIndexed=BASE_PARAMS["queryFromIndexed"],
                queryNoiseEps=noise,
            )
            runs_done += 1
        except subprocess.CalledProcessError as e:
            print(f"Run failed with error: {e}. Continuing with next combination.")


# ============================================================
# CSV loading & compact summary
# ============================================================

def load_runs_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # avgQueryTime = queryTime / nQueries
    if "queryTime" in df.columns and "nQueries" in df.columns:
        df["avgQueryTime"] = df["queryTime"] / df["nQueries"].replace(0, pd.NA)
    else:
        df["avgQueryTime"] = pd.NA

    return df


def export_compact_metrics(df: pd.DataFrame, out_path: Path):
    """
    Φτιάχνει ένα extra CSV μόνο με τις βασικές μετρήσεις που μας νοιάζουν.
    """
    cols_we_want = [
        "algorithmType",
        "dataPath",
        "datasetName",
        "normalize",
        "queryNoiseEps",
        "K",
        "nQueries",
        "queryTime",
        "avgQueryTime",
        "indexBuildTime",
        "memoryUsageBeforeIndexing",
        "memoryUsageAfterIndexing",
        "indexSize",
        "N",
        "channels",
        "nQueryChannels",
    ]

    existing_cols = [c for c in cols_we_want if c in df.columns]
    compact = df[existing_cols].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(out_path, index=False)
    print(f"Saved compact metrics CSV: {out_path}")


# ============================================================
# Plotting
# ============================================================

def plot_query_time_vs_nqueries(df: pd.DataFrame, out_dir: Path):
    """
    Για κάθε dataset + normalize, plot avgQueryTime vs nQueries για κάθε algorithmType.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_f = df[df["algorithmType"].isin(ALGORITHMS)].copy()
    df_f = df_f.dropna(subset=["avgQueryTime"])

    for dataset in df_f["dataPath"].unique():
        for norm in df_f["normalize"].unique():
            sub = df_f[(df_f["dataPath"] == dataset) &
                       (df_f["normalize"] == norm)]

            if sub.empty:
                continue

            plt.figure()
            for alg in ALGORITHMS:
                sub_alg = sub[sub["algorithmType"] == alg]
                if sub_alg.empty:
                    continue

                sub_alg = sub_alg.sort_values("nQueries")
                plt.plot(
                    sub_alg["nQueries"],
                    sub_alg["avgQueryTime"],
                    marker="o",
                    label=alg,
                )

            plt.xscale("log")
            plt.xlabel("nQueries (log scale)")
            plt.ylabel("avgQueryTime (ms per query)")
            plt.title(f"avgQueryTime vs nQueries\n{dataset}, normalize={norm}")
            plt.legend()
            plt.tight_layout()

            dataset_label = str(dataset)
            fname = out_dir / f"avgQueryTime_{dataset_label}_norm_{norm}.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")


def plot_index_time_and_memory(df: pd.DataFrame, out_dir: Path):
    """
    Bar plots για indexBuildTime και memoryUsageAfterIndexing
    ανά algorithmType (grouped ανά dataset + normalize).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_f = df[df["algorithmType"].isin(ALGORITHMS)].copy()

    for dataset in df_f["dataPath"].unique():
        for norm in df_f["normalize"].unique():
            sub = df_f[(df_f["dataPath"] == dataset) &
                       (df_f["normalize"] == norm)]

            if sub.empty:
                continue

            agg = sub.groupby("algorithmType").agg({
                "indexBuildTime": "mean",
                "memoryUsageAfterIndexing": "mean",
            }).reset_index()

            # Index build time
            plt.figure()
            plt.bar(agg["algorithmType"], agg["indexBuildTime"])
            plt.ylabel("indexBuildTime (ms, mean)")
            plt.title(f"Index build time per algorithm\n{dataset}, normalize={norm}")
            plt.tight_layout()
            fname1 = out_dir / f"indexBuildTime_{Path(dataset).name}_norm_{norm}.png"
            plt.savefig(fname1, dpi=200)
            plt.close()
            print(f"Saved plot: {fname1}")

            # Memory usage
            plt.figure()
            plt.bar(agg["algorithmType"], agg["memoryUsageAfterIndexing"])
            plt.ylabel("memoryUsageAfterIndexing (MB, mean)")
            plt.title(f"Memory usage per algorithm\n{dataset}, normalize={norm}")
            plt.tight_layout()
            fname2 = out_dir / f"memoryUsage_{Path(dataset).name}_norm_{norm}.png"
            plt.savefig(fname2, dpi=200)
            plt.close()
            print(f"Saved plot: {fname2}")


def plot_query_time_vs_noise(df: pd.DataFrame, out_dir: Path):
    """
    Για κάθε dataset + normalize:
    x-axis: queryNoiseEps
    y-axis: mean avgQueryTime
    μία γραμμή ανά algorithmType (MSINDEX, MSINDEXPP, MASS).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_f = df[df["algorithmType"].isin(ALGORITHMS)].copy()
    df_f = df_f.dropna(subset=["avgQueryTime"])

    grouped = (
        df_f
        .groupby(["dataPath", "normalize", "algorithmType", "queryNoiseEps"])
        .agg(avgQueryTime_mean=("avgQueryTime", "mean"))
        .reset_index()
    )

    for dataset in grouped["dataPath"].unique():
        for norm in grouped["normalize"].unique():
            sub = grouped[
                (grouped["dataPath"] == dataset) &
                (grouped["normalize"] == norm)
            ]
            if sub.empty:
                continue

            plt.figure()
            for alg in ALGORITHMS:
                sub_alg = sub[sub["algorithmType"] == alg]
                if sub_alg.empty:
                    continue

                sub_alg = sub_alg.sort_values("queryNoiseEps")
                plt.plot(
                    sub_alg["queryNoiseEps"],
                    sub_alg["avgQueryTime_mean"],
                    marker="o",
                    label=alg,
                )

            plt.xlabel("queryNoiseEps")
            plt.ylabel("avg avgQueryTime (ms per query)")
            plt.title(
                f"avgQueryTime vs noise\n"
                f"{dataset}, normalize={norm}"
            )
            plt.legend()
            plt.tight_layout()

            fname = out_dir / f"avgQueryTime_vs_noise_{Path(dataset).name}_norm_{norm}.png"
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"Saved plot: {fname}")


# ============================================================
# Main
# ============================================================

def main():
    project_root = Path(__file__).resolve().parent
    runs_csv = project_root / "output" / "runs.csv"
    plots_dir = project_root / "plots"
    summary_csv = project_root / "output" / "summary_metrics.csv"

    if RUN_EXPERIMENTS:
        run_all_experiments()

    if MAKE_PLOTS:
        df = load_runs_csv(runs_csv)
        print(f"Loaded {len(df)} rows from {runs_csv}")

        # Extra compact CSV με τις βασικές μετρήσεις
        export_compact_metrics(df, summary_csv)

        # Plots
        plot_query_time_vs_nqueries(df, plots_dir / "query_time")
        plot_index_time_and_memory(df, plots_dir / "index_and_memory")
        plot_query_time_vs_noise(df, plots_dir / "noise")


if __name__ == "__main__":
    main()
