#!/usr/bin/env python3.10

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List
from functools import lru_cache


ROOT = Path(__file__).resolve().parents[1]
JAR_PATH = ROOT / "target" / "MS-Index-1.0-jar-with-dependencies.jar"
TRAINER_PATH = ROOT / "scripts" / "train_msindexpp_gbdt.py"
ARTIFACT_ROOT = ROOT / "artifacts" / "msindexpp_suite"

K_VALUES = [1, 10, 100]
NORMALIZE_VALUES = [False, True]
FROM_INDEX_VALUES = [False, True]
NOISE_VALUES = [round(v * 0.1, 1) for v in range(10)]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str
    q_len: int
    default_n: int = 100
    max_len: int | None = None
    channels: int = -1


# Built-in dataset shortcuts — not used when --dataset-spec is provided (as in run_suite.sh).
DATASETS: Dict[str, DatasetConfig] = {}


def parse_dataset_spec(spec: str) -> DatasetConfig:
    parts = spec.split(":")
    if len(parts) not in (3, 4, 5):
        raise ValueError(f"Invalid --dataset-spec '{spec}'. Expected name:path:qLen[:defaultN[:maxLen]].")
    name, path, q_len = parts[:3]
    default_n = int(parts[3]) if len(parts) >= 4 else 100
    max_len = int(parts[4]) if len(parts) == 5 else None
    return DatasetConfig(name=name, path=path, q_len=int(q_len), default_n=default_n, max_len=max_len)


def resolve_datasets(args) -> List[DatasetConfig]:
    datasets = []
    seen = set()
    for dataset_name in args.datasets:
        dataset = DATASETS[dataset_name]
        if dataset.name not in seen:
            datasets.append(dataset)
            seen.add(dataset.name)
    for spec in args.dataset_spec:
        dataset = parse_dataset_spec(spec)
        if dataset.name not in seen:
            datasets.append(dataset)
            seen.add(dataset.name)
    return datasets


def combo_key(k: int, normalize: bool, from_index: bool, noise: float) -> str:
    return "_".join(
        [
            f"k{k}",
            f"norm{int(normalize)}",
            f"from{int(from_index)}",
            f"noise{noise:.1f}".replace(".", "p"),
        ]
    )


def run_command(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(cmd)}. See {log_path}")


def parse_single_run_csv(run_dir: Path) -> Dict[str, str]:
    runs_csv = run_dir / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"Expected {runs_csv}")
    with runs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {runs_csv}")
    return rows[-1]


def ensure_package() -> None:
    if JAR_PATH.exists():
        return
    run_command(["mvn", "-q", "-DskipTests", "package"], ROOT / "artifacts" / "msindexpp_suite" / "build.log")


def clear_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def java_base_cmd(output_dir: Path, extra_props: Dict[str, str], algorithm: str, dataset: DatasetConfig,
                  n: int, max_m: int, k: int, normalize: bool, n_queries: int,
                  experiment_id: int, seed: int, query_from_index: bool, noise: float,
                  runtime_mode: str) -> List[str]:
    cmd = ["java", f"-Doutput.path={output_dir}"]
    for key, value in extra_props.items():
        cmd.append(f"-D{key}={value}")
    cmd.extend(
        [
            "-cp",
            str(JAR_PATH),
            "io/github/Main",
            algorithm,
            dataset.path,
            str(n),
            str(max_m),
            str(dataset.channels),
            "-1",
            str(dataset.q_len),
            str(k),
            str(normalize).lower(),
            str(n_queries),
            runtime_mode,
            str(experiment_id),
            str(seed),
            "false",
            str(query_from_index).lower(),
            str(noise),
        ]
    )
    return cmd


def effective_n(args, dataset: DatasetConfig) -> int:
    return dataset.default_n if args.n is None else args.n


@lru_cache(maxsize=None)
def count_holdout_eligible_series(dataset_path: str, q_len: int) -> int:
    dataset_root = Path(dataset_path)
    variate_paths = sorted(dataset_root.glob("dim_*"))
    if not variate_paths:
        return 0

    eligible = 0
    min_required_length = q_len * 4
    with variate_paths[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if len(line.split(",")) >= min_required_length:
                eligible += 1
    return eligible


def supported_from_index_values(dataset: DatasetConfig, args) -> List[bool]:
    if getattr(args, "from_index_values", None) is not None:
        return args.from_index_values
    if dataset.max_len is not None and dataset.max_len < dataset.q_len * 4:
        return [True]
    required_holdout_series = min(effective_n(args, dataset), args.n_queries)
    eligible_series = count_holdout_eligible_series(dataset.path, dataset.q_len)
    if eligible_series < required_holdout_series:
        return [True]
    return FROM_INDEX_VALUES


def noise_values_for(from_index: bool, args) -> List[float]:
    # Hold-out OOD queries are sampled from withheld series with zero added
    # noise by default. Extra OOD noise values may be requested explicitly.
    return args.ood_noise_values if not from_index else NOISE_VALUES


def maybe_run_java(stage_dir: Path, log_name: str, cmd: List[str]) -> Dict[str, str]:
    runs_csv = stage_dir / "runs.csv"
    if not runs_csv.exists():
        run_command(cmd, stage_dir / log_name)
    return parse_single_run_csv(stage_dir)


def ensure_index_built(dataset: DatasetConfig, algorithm: str, normalize: bool, args, experiment_id: int) -> int:
    stage_dir = ARTIFACT_ROOT / dataset.name / "index_builds" / algorithm.lower() / f"norm{int(normalize)}"
    runs_csv = stage_dir / "runs.csv"
    if not runs_csv.exists():
        cmd = java_base_cmd(
            stage_dir,
            {},
            algorithm,
            dataset,
            effective_n(args, dataset),
            args.max_m,
            1,
            normalize,
            1,
            experiment_id,
            args.seed,
            True,
            0.0,
            "INDEX",
        )
        run_command(cmd, stage_dir / "build.log")
        experiment_id += 1
    return experiment_id


def parse_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def parse_int(row: Dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value != "" else 0


def parse_query_time_avg_ms(row: Dict[str, str]) -> float:
    per_query = row.get("queryTimePerQuery", "")
    if per_query:
        values = [float(value) for value in per_query.split(";") if value]
        if values:
            # Ignore the first query when multiple measurements exist: it can
            # include one-time JVM/model warmup such as lazy MSIndex++ model load.
            steady_state_values = values[1:] or values
            return sum(steady_state_values) / len(steady_state_values)

    n_queries = parse_int(row, "nQueries")
    query_time = parse_float(row, "queryTime")
    return query_time / n_queries if n_queries > 0 else query_time


def run_stage_tasks(tasks: List[Dict], worker_count: int) -> List[Dict]:
    if worker_count <= 1:
        return [task["fn"](**task["kwargs"]) for task in tasks]

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(task["fn"], **task["kwargs"]): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return results


def run_training_combo(dataset: DatasetConfig, args, dataset_root: Path, train_data_part_path: Path,
                       k: int, normalize: bool, from_index: bool, noise: float, experiment_id: int) -> Dict:
    key = combo_key(k, normalize, from_index, noise)
    stage_dir = dataset_root / "train_runs" / key
    cache_hit = (stage_dir / "runs.csv").exists()
    extra_props = {
        "msindexpp.trainingEnabled": "true",
        "msindexpp.checkpointInterval": str(args.checkpoint_interval),
        "msindexpp.accuracyTarget": str(args.accuracy_target),
        "msindexpp.artifactsDir": str(dataset_root),
        "msindexpp.trainingDataPath": str(train_data_part_path),
    }
    stage_start = time.monotonic()
    row = maybe_run_java(
        stage_dir,
        "train.log",
        java_base_cmd(
            stage_dir,
            extra_props,
            "MSINDEXPP",
            dataset,
            effective_n(args, dataset),
            args.max_m,
            k,
            normalize,
            args.n_queries,
            experiment_id,
            args.seed,
            from_index,
            noise,
            "QUERY",
        ),
    )
    stage_wall_time_s = time.monotonic() - stage_start
    return {
        "dataset": dataset.name,
        "combo": key,
        "experiment_id": experiment_id,
        "k": k,
        "normalize": normalize,
        "from_index": from_index,
        "noise": noise,
        "query_time_ms": parse_query_time_avg_ms(row),
        "query_time_total_ms": parse_float(row, "queryTime"),
        "index_build_time_ms": parse_float(row, "indexBuildTime"),
        "segments_under_threshold": parse_float(row, "segmentsUnderThreshold"),
        "training_part_path": str(train_data_part_path),
        "stage_wall_time_s": stage_wall_time_s,
        "cache_hit": cache_hit,
    }


def run_eval_combo(dataset: DatasetConfig, args, dataset_root: Path, model_path: Path,
                   k: int, normalize: bool, from_index: bool, noise: float, experiment_id: int) -> Dict:
    key = combo_key(k, normalize, from_index, noise)
    stage_dir = dataset_root / "eval_runs" / key
    extra_props = {
        "msindexpp.earlyTerminationEnabled": "true",
        "msindexpp.evaluationEnabled": "true",
        "msindexpp.checkpointInterval": str(args.checkpoint_interval),
        "msindexpp.modelScoreThreshold": str(args.model_score_threshold),
        "msindexpp.artifactsDir": str(dataset_root),
        "msindexpp.modelPath": str(model_path),
    }
    row = maybe_run_java(
        stage_dir,
        "eval.log",
        java_base_cmd(
            stage_dir,
            extra_props,
            "MSINDEXPP",
            dataset,
            effective_n(args, dataset),
            args.max_m,
            k,
            normalize,
            args.n_queries,
            experiment_id,
            args.seed,
            from_index,
            noise,
            "QUERY",
        ),
    )
    return {
        "dataset": dataset.name,
        "combo": key,
        "experiment_id": experiment_id,
        "k": k,
        "normalize": normalize,
        "from_index": from_index,
        "noise": noise,
        "avg_accuracy": parse_float(row, "msIndexPPEarlyTerminationAccuracySum") / args.n_queries,
        "early_stop_rate": parse_float(row, "msIndexPPEarlyStops") / args.n_queries,
        "avg_early_steps": parse_float(row, "msIndexPPEarlyTerminationSteps")
        / max(parse_int(row, "msIndexPPEarlyStops"), 1),
        "query_time_ms": parse_query_time_avg_ms(row),
        "query_time_total_ms": parse_float(row, "queryTime"),
    }


def run_timing_combo(dataset: DatasetConfig, args, dataset_root: Path, model_path: Path,
                     algorithm: str, k: int, normalize: bool, from_index: bool, noise: float,
                     experiment_id: int) -> Dict:
    key = combo_key(k, normalize, from_index, noise)
    stage_dir = dataset_root / "timing_runs" / algorithm.lower() / key
    runtime_mode = "FULL_NO_STORE" if algorithm == "MASS" else "QUERY"
    extra_props = {}
    if algorithm == "MSINDEXPP":
        extra_props = {
            "msindexpp.earlyTerminationEnabled": "true",
            "msindexpp.checkpointInterval": str(args.checkpoint_interval),
            "msindexpp.modelScoreThreshold": str(args.model_score_threshold),
            "msindexpp.artifactsDir": str(dataset_root),
            "msindexpp.modelPath": str(model_path),
        }
    row = maybe_run_java(
        stage_dir,
        f"{algorithm.lower()}.log",
        java_base_cmd(
            stage_dir,
            extra_props,
            algorithm,
            dataset,
            effective_n(args, dataset),
            args.max_m,
            k,
            normalize,
            args.n_queries,
            experiment_id,
            args.seed,
            from_index,
            noise,
            runtime_mode,
        ),
    )
    return {
        "dataset": dataset.name,
        "algorithm": algorithm,
        "combo": key,
        "experiment_id": experiment_id,
        "k": k,
        "normalize": normalize,
        "from_index": from_index,
        "noise": noise,
        "query_time_ms": parse_query_time_avg_ms(row),
        "query_time_total_ms": parse_float(row, "queryTime"),
        "index_build_time_ms": parse_float(row, "indexBuildTime"),
        "index_search_time_ms": parse_float(row, "indexSearchTime"),
        "exhaustive_time_ms": parse_float(row, "exhaustiveTime"),
    }


def train_dataset_model(dataset: DatasetConfig, args, experiment_start: int) -> int:
    dataset_train_start = time.monotonic()
    dataset_root = ARTIFACT_ROOT / dataset.name
    train_data_path = dataset_root / "training-data.jsonl"
    model_path = dataset_root / "gbdt-model.json"
    trainer_metrics_path = dataset_root / "trainer_metrics.txt"
    train_results_path = dataset_root / "train_collection_results.jsonl"
    training_runtime_path = dataset_root / "training_runtime.json"
    training_parts_dir = dataset_root / "training_parts"
    train_runs_dir = dataset_root / "train_runs"

    if args.force:
        clear_file(train_data_path)
        clear_file(model_path)
        clear_file(trainer_metrics_path)
        clear_file(train_results_path)
        clear_file(training_runtime_path)
        if training_parts_dir.exists():
            shutil.rmtree(training_parts_dir)
        if train_runs_dir.exists():
            shutil.rmtree(train_runs_dir)

    dataset_root.mkdir(parents=True, exist_ok=True)
    training_parts_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = experiment_start
    for normalize in NORMALIZE_VALUES:
        experiment_id = ensure_index_built(dataset, "MSINDEXPP", normalize, args, experiment_id)
    tasks = []
    combo_rows = []
    for k, normalize, from_index in product(K_VALUES, NORMALIZE_VALUES, supported_from_index_values(dataset, args)):
        for noise in noise_values_for(from_index, args):
            combo_rows.append((k, normalize, from_index, noise))
    for k, normalize, from_index, noise in combo_rows:
        key = combo_key(k, normalize, from_index, noise)
        training_part_path = training_parts_dir / f"{key}.jsonl"
        tasks.append(
            {
                "fn": run_training_combo,
                "kwargs": {
                    "dataset": dataset,
                    "args": args,
                    "dataset_root": dataset_root,
                    "train_data_part_path": training_part_path,
                    "k": k,
                    "normalize": normalize,
                    "from_index": from_index,
                    "noise": noise,
                    "experiment_id": experiment_id,
                },
            }
        )
        experiment_id += 1
    collection_start = time.monotonic()
    train_records = run_stage_tasks(tasks, args.max_workers)
    collection_wall_time_s = time.monotonic() - collection_start

    merge_start = time.monotonic()
    with train_data_path.open("w", encoding="utf-8") as merged:
        for record in sorted(train_records, key=lambda item: item["combo"]):
            training_part_path = Path(record["training_part_path"])
            if training_part_path.exists():
                merged.write(training_part_path.read_text(encoding="utf-8"))

    with train_results_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record) + "\n")
    merge_wall_time_s = time.monotonic() - merge_start

    trainer_start = time.monotonic()
    run_command(
        [
            "/usr/bin/python3.10",
            str(TRAINER_PATH),
            "--input",
            str(train_data_path),
            "--output",
            str(model_path),
        ],
        trainer_metrics_path,
    )
    trainer_wall_time_s = time.monotonic() - trainer_start

    training_runtime = {
        "dataset": dataset.name,
        "num_training_combos": len(train_records),
        "max_workers": args.max_workers,
        "collection_wall_time_s": collection_wall_time_s,
        "collection_stage_wall_time_sum_s": sum(record["stage_wall_time_s"] for record in train_records),
        "collection_cache_hits": sum(1 for record in train_records if record["cache_hit"]),
        "merge_wall_time_s": merge_wall_time_s,
        "trainer_wall_time_s": trainer_wall_time_s,
        "total_wall_time_s": time.monotonic() - dataset_train_start,
        "training_data_bytes": train_data_path.stat().st_size if train_data_path.exists() else 0,
        "model_bytes": model_path.stat().st_size if model_path.exists() else 0,
    }
    training_runtime_path.write_text(json.dumps(training_runtime, indent=2) + "\n", encoding="utf-8")
    return experiment_id


def evaluate_dataset_model(dataset: DatasetConfig, args, experiment_start: int) -> int:
    dataset_root = ARTIFACT_ROOT / dataset.name
    model_path = dataset_root / "gbdt-model.json"
    eval_results_path = dataset_root / "evaluation_results.jsonl"
    eval_runs_dir = dataset_root / "eval_runs"
    experiment_id = experiment_start
    for normalize in NORMALIZE_VALUES:
        experiment_id = ensure_index_built(dataset, "MSINDEXPP", normalize, args, experiment_id)
    tasks = []

    if args.force:
        clear_file(eval_results_path)
        if eval_runs_dir.exists():
            shutil.rmtree(eval_runs_dir)

    for k, normalize, from_index in product(K_VALUES, NORMALIZE_VALUES, supported_from_index_values(dataset, args)):
        for noise in noise_values_for(from_index, args):
            tasks.append(
                {
                    "fn": run_eval_combo,
                    "kwargs": {
                        "dataset": dataset,
                        "args": args,
                        "dataset_root": dataset_root,
                        "model_path": model_path,
                        "k": k,
                        "normalize": normalize,
                        "from_index": from_index,
                        "noise": noise,
                        "experiment_id": experiment_id,
                    },
                }
            )
            experiment_id += 1
    eval_records = run_stage_tasks(tasks, args.max_workers)

    with eval_results_path.open("w", encoding="utf-8") as handle:
        for record in eval_records:
            handle.write(json.dumps(record) + "\n")
    return experiment_id


def benchmark_dataset(dataset: DatasetConfig, args, experiment_start: int) -> int:
    dataset_root = ARTIFACT_ROOT / dataset.name
    model_path = dataset_root / "gbdt-model.json"
    timing_results_path = dataset_root / "timing_results.jsonl"
    experiment_id = experiment_start
    records = []

    if args.force:
        clear_file(timing_results_path)

    algorithm_configs = {
        "MSINDEX": ("MSINDEX", {}),
        "MASS": ("MASS", {}),
        "MSINDEXPP": (
            "MSINDEXPP",
            {
                "msindexpp.earlyTerminationEnabled": "true",
                "msindexpp.checkpointInterval": str(args.checkpoint_interval),
                "msindexpp.modelScoreThreshold": str(args.model_score_threshold),
                "msindexpp.artifactsDir": str(dataset_root),
                "msindexpp.modelPath": str(model_path),
            },
        ),
    }
    algorithms = [algorithm_configs[name] for name in args.benchmark_algorithms]

    for normalize in NORMALIZE_VALUES:
        if "MSINDEX" in args.benchmark_algorithms:
            experiment_id = ensure_index_built(dataset, "MSINDEX", normalize, args, experiment_id)
        if "MSINDEXPP" in args.benchmark_algorithms and "MSINDEX" not in args.benchmark_algorithms:
            experiment_id = ensure_index_built(dataset, "MSINDEXPP", normalize, args, experiment_id)

    tasks = []
    for k, normalize, from_index in product(K_VALUES, NORMALIZE_VALUES, supported_from_index_values(dataset, args)):
        for noise in noise_values_for(from_index, args):
            key = combo_key(k, normalize, from_index, noise)
            for algorithm, extra_props in algorithms:
                tasks.append(
                    {
                        "fn": run_timing_combo,
                        "kwargs": {
                            "dataset": dataset,
                            "args": args,
                            "dataset_root": dataset_root,
                            "model_path": model_path,
                            "algorithm": algorithm,
                            "k": k,
                            "normalize": normalize,
                            "from_index": from_index,
                            "noise": noise,
                            "experiment_id": experiment_id,
                        },
                    }
                )
                experiment_id += 1
    records = run_stage_tasks(tasks, args.max_workers)
    if timing_results_path.exists() and not args.force:
        merged_records = load_jsonl(timing_results_path)
        record_index = {
            (
                record.get("algorithm"),
                record.get("combo"),
                record.get("k"),
                record.get("normalize"),
                record.get("from_index"),
                record.get("noise"),
            ): index
            for index, record in enumerate(merged_records)
        }
        for record in records:
            key = (
                record.get("algorithm"),
                record.get("combo"),
                record.get("k"),
                record.get("normalize"),
                record.get("from_index"),
                record.get("noise"),
            )
            if key in record_index:
                merged_records[record_index[key]] = record
            else:
                record_index[key] = len(merged_records)
                merged_records.append(record)
        records = merged_records

    with timing_results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return experiment_id


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_dataset(dataset: str) -> Dict[str, Dict]:
    dataset_root = ARTIFACT_ROOT / dataset
    eval_rows = load_jsonl(dataset_root / "evaluation_results.jsonl")
    timing_rows = load_jsonl(dataset_root / "timing_results.jsonl")
    training_runtime_path = dataset_root / "training_runtime.json"

    summary = {"dataset": dataset}
    if training_runtime_path.exists():
        training_runtime = json.loads(training_runtime_path.read_text(encoding="utf-8"))
        summary["training_total_wall_time_s"] = training_runtime.get("total_wall_time_s", 0.0)
        summary["training_collection_wall_time_s"] = training_runtime.get("collection_wall_time_s", 0.0)
        summary["training_trainer_wall_time_s"] = training_runtime.get("trainer_wall_time_s", 0.0)
        summary["training_num_combos"] = training_runtime.get("num_training_combos", 0)
    if eval_rows:
        summary["accuracy_avg"] = sum(r["avg_accuracy"] for r in eval_rows) / len(eval_rows)
        summary["accuracy_min"] = min(r["avg_accuracy"] for r in eval_rows)
        summary["accuracy_max"] = max(r["avg_accuracy"] for r in eval_rows)
        summary["early_stop_rate_avg"] = sum(r["early_stop_rate"] for r in eval_rows) / len(eval_rows)
        summary["avg_early_steps"] = sum(r["avg_early_steps"] for r in eval_rows) / len(eval_rows)

    for algorithm in ["MSINDEX", "MASS", "MSINDEXPP"]:
        rows = [r for r in timing_rows if r["algorithm"] == algorithm]
        if rows:
            prefix = algorithm.lower()
            summary[f"{prefix}_query_time_avg_ms"] = sum(r["query_time_ms"] for r in rows) / len(rows)
            summary[f"{prefix}_query_time_min_ms"] = min(r["query_time_ms"] for r in rows)
            summary[f"{prefix}_query_time_max_ms"] = max(r["query_time_ms"] for r in rows)
            summary[f"{prefix}_index_build_avg_ms"] = sum(r["index_build_time_ms"] for r in rows) / len(rows)

    msindex_avg = summary.get("msindex_query_time_avg_ms")
    mass_avg = summary.get("mass_query_time_avg_ms")
    early_avg = summary.get("msindexpp_query_time_avg_ms")
    if msindex_avg and early_avg:
        summary["speedup_vs_msindex"] = msindex_avg / early_avg if early_avg else 0.0
    if mass_avg and early_avg:
        summary["speedup_vs_mass"] = mass_avg / early_avg if early_avg else 0.0
    return summary


def main() -> int:
    global ARTIFACT_ROOT
    parser = argparse.ArgumentParser(description="Run MSIndexPP train/eval/benchmark suite per dataset.")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--dataset-spec", action="append", default=[], help="Custom dataset as name:path:qLen[:defaultN]. May be provided multiple times.")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--n", type=int, default=None, help="Number of time series to load per dataset. If omitted, each dataset uses its default.")
    parser.add_argument("--max-m", type=int, default=100000)
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=64)
    parser.add_argument("--accuracy-target", type=float, default=0.9)
    parser.add_argument("--model-score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--normalize-values",
        nargs="+",
        choices=("false", "true"),
        default=None,
        help="Restrict the run to raw/normalized data. Default: both values.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--benchmark-algorithms",
        nargs="+",
        choices=("MSINDEX", "MASS", "MSINDEXPP"),
        default=["MSINDEX", "MASS", "MSINDEXPP"],
        help="Algorithms to run during benchmark/timing. Use MSINDEX only for exact L2 reference logs.",
    )
    parser.add_argument("--force", action="store_true", help="Delete cached stage outputs and rerun.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--from-index-values",
        nargs="+",
        choices=("0", "1", "false", "true"),
        default=None,
        help="Restrict runs to specific queryFromIndexed values. Example: --from-index-values false",
    )
    parser.add_argument(
        "--ood-noise-values",
        nargs="+",
        type=float,
        default=[0.0],
        help="Noise values to run for queryFromIndexed=false OOD queries.",
    )
    args = parser.parse_args()
    global NORMALIZE_VALUES
    if args.normalize_values is not None:
        NORMALIZE_VALUES = [value == "true" for value in args.normalize_values]
    if args.from_index_values is not None:
        args.from_index_values = [value in ("1", "true") for value in args.from_index_values]

    ARTIFACT_ROOT = args.artifact_root
    ensure_package()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    experiment_id = 1000
    summaries = []
    datasets = resolve_datasets(args)

    for dataset in datasets:
        if not Path(dataset.path).exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset.path}")

        if not args.skip_train:
            experiment_id = train_dataset_model(dataset, args, experiment_id)
        if not args.skip_eval:
            experiment_id = evaluate_dataset_model(dataset, args, experiment_id)
        if not args.skip_benchmark:
            experiment_id = benchmark_dataset(dataset, args, experiment_id)

        summaries.append(summarize_dataset(dataset.name))

    write_summary_csv(
        ARTIFACT_ROOT / "dataset_summary.csv",
        summaries,
        sorted({key for row in summaries for key in row.keys()}),
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
