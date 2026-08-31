# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase implementing **MS-Index** (and extensions **MSIndex++** and **HYBRID**), algorithms for fast
subsequence search in multivariate time series under Euclidean distance. It's a Maven/Java 17 project with
Python (scikit-learn/numpy) scripts bolted on for training GBDT models used by two of the algorithms at query time.
The repo is run interactively and via Slurm batch jobs on an HPC cluster (paths like `/sw/arch/RHEL9/...` and
`#SBATCH` headers in `slurm/` and `scripts/*_jobs.sh` reflect that environment) — don't assume a generic Linux box.

There are no unit tests in this repo; correctness is checked at runtime via `runtimeMode=CORRECTNESS`, which
compares index results against brute force per query (see `Algorithm.queryAll()`).

## Build & run

```bash
mvn -q -DskipTests package        # builds target/MS-Index-1.0-jar-with-dependencies.jar (main: io.github.Main)
./run.sh                          # edit the variables at the top of run.sh, then runs the jar
```

Direct invocation (positional args, see `Main.setParameters`):

```bash
java -cp target/MS-Index-1.0-jar-with-dependencies.jar io/github/Main \
  <algorithmType> <dataPath> <N> <maxM> <channels> <nQueryChannels> <qLen> <K> \
  <normalize> <nQueries> <runtimeMode> <experimentId> <seed> <parallel> \
  <queryFromIndexed> <queryNoiseEps>
```

- `algorithmType`: `MSINDEX`, `MSINDEXPP`, `HYBRID`, `BRUTE_FORCE`, `MASS`, `ST_INDEX`, `DSTREE`, `KV_MATCH`
- `runtimeMode`: `FULL`, `FULL_NO_STORE`, `INDEX`, `INDEX_NO_STORE`, `QUERY`, `CHECK_INDEX`, `CORRECTNESS`
- Passing no args at all runs with the hardcoded defaults in `Main.setParameters`, not `run.sh`'s values.

MSIndexPP/HYBRID-specific behavior is controlled via `-D` JVM system properties layered on top of the positional
args (see `Parameters.applySystemOverrides()`), e.g. `-Dmsindexpp.trainingEnabled=true`,
`-Dmsindexpp.earlyTerminationEnabled=true`, `-Dhybrid.trainingCsv=...`. The README has the full property list and
worked examples for both — read it before touching either feature.

Python side (GBDT training/eval), under a local `.venv` (Python 3.9, numpy + scikit-learn required):

```bash
python3 scripts/train_msindexpp_gbdt.py --input <training-data.jsonl> --output <gbdt-model.json>
python3 scripts/train_hybrid_gbdt.py ...
python3 scripts/calibrate_conformal.py ...
```

## Experiment orchestration (`scripts/`)

Everything reads dataset paths and shared knobs (queries per experiment, worker count, seed) from
`scripts/config.sh` — edit that file before running any suite, not the individual job scripts. Several scripts
also honor environment-variable overrides of the same names (`UEA_BASE_DIR`, `N_QUERIES`, `JAR_PATH`,
`ARTIFACT_ROOT`, etc.) captured before `config.sh` is sourced.

- `run_suite.sh` — runs the full MSINDEX/MASS/MSINDEXPP benchmark across all 34 datasets (or `--datasets custom`
  for just the synthetic/weather/stocks/wind ones) via `run_msindexpp_dataset_suite.py`. Writes to
  `artifacts/msindexpp_suite/<dataset>/` plus a combined `dataset_summary.csv`.
- `EARLY_TERMINATION_jobs.sh` — per-dataset MSIndexPP early-termination training + evaluation, three model
  variants (raw/normalized/joint). Supports `--print-commands`, `--submit` (one Slurm job per dataset), or
  `--dataset <name>` to run one directly. Training data stays under `<artifacts_early_termination>/training_private`;
  models/eval results are mirrored to `.../results` so results can be shared without shipping training data.
- `HYBRID_jobs.sh` — same shape as above but for the MASS-vs-MSINDEX routing model (`artifacts_hybrid/`).
- `E1_E2_jobs.sh`, `E1_E2_retry_timeouts.sh`, `E3_jobs.sh` — the paper's numbered experiment suites; the retry
  script re-runs specifically the combos that previously timed out.
- Build the jar once (`mvn -q -DskipTests package`) before running any of these; they check for it and build it
  themselves if missing, but that first build is slow so don't rely on it happening silently mid-batch-submit.

`slurm/` holds the raw sbatch templates (module loads, `-Xmx`, `ForkJoinPool` parallelism) that the `*_jobs.sh`
scripts fill in and submit when given `--submit`.

## Architecture

**Entry point:** `io.github.Main` parses parameters into static fields on `io.github.utils.Parameters`
(everything downstream reads these statics rather than passing config objects around — this is the pattern to
follow when adding a new tunable), loads/generates the dataset via `DataLoader`/`DataManager`, builds a workload
via `WorkloadGenerator`, dispatches to the selected algorithm, and dumps parameters + stats via
`Parameters.saveParams()` (appends a row to `output/runs.csv`).

**Algorithm abstraction** (`io.github.algorithms.Algorithm`, abstract class, not an interface): subclasses
implement `buildIndex()` and `kNN(k, query)`. The base class's `run()` drives the index/save/load/query sequence
according to `runtimeMode`, and `queryAll()` times each query and (in `CORRECTNESS` mode) diffs against
`BruteForce`. Index files are cached under `index_caches/*.ser`, keyed by a filename encoding every parameter that
affects the index (algorithm, dataset, N, maxM, seed, qLen, channels, normalize, fftCoveredDistance, kMeansClusters)
— see `Algorithm.getIndexPath()`. If you add a new index-affecting parameter, add it to that key or you'll get
stale-index cache hits.

**Algorithm implementations** (`io.github.algorithms/`):
- `msindex/` (`MSIndex`) — the paper's core index: R-tree-style MBRs over FFT-based segment sketches
  (`msindex/segmentation/*`: `AdhocSegment`, `EquiDepthSegment`, `EquiLengthSegment`, `FalSegment`, `GreedySegment`,
  `NoSegment`, `SingletonSegment` — pluggable segmentation strategies).
- `msindexpp/` (`MSIndexPP`) — MSIndex plus a learned early-termination policy for the second (exhaustive
  verification) pass. `msindexpp/ml/`: `SecondPassTraversal` walks candidates in checkpoints, `TraversalFeatureSnapshot`
  captures features at each checkpoint, `TrainingDataWriter` emits the labeled JSONL rows Python trains on, and
  `PortableGBDTModel` is the Java-side inference engine for the model JSON exported by
  `scripts/train_msindexpp_gbdt.py`. `msindex/` and `msindexpp/` are near-duplicates of each other (and of
  `msindexshared/`, a partially-factored-out common base) — check all three when changing shared MBR/segmentation
  logic, since fixes don't automatically propagate between them.
- `hybrid/` (`Hybrid`) — builds both MASS and MSIndex, extracts a fixed feature vector per query
  (`QueryDecisionFeatureExtractor` / `DecisionFeatureVector`), and uses a `PortableGBDTDecisionModel` (trained by
  `scripts/train_hybrid_gbdt.py`) to route each query to whichever exact method it predicts is faster.
  `DecisionTrainingCsvWriter` / `DecisionEvaluationCsvWriter` handle the CSV logging for training/eval passes.
- `mass/` (`MVMASS`) — brute-force-equivalent baseline via Mueen's MASS algorithm (FFT-based distance profile).
- `dstree_org/`, `stindex/`, `kvmatch/` — ported baseline index structures (DSTree, ST-Index, KV-match) used for
  comparison in experiments; treat as mostly-frozen third-party-style code rather than code to extend.
- `BruteForce` — the ground-truth reference implementation used by `CORRECTNESS` mode and by `Algorithm.postFilter`.

**Shared utilities** (`io.github.utils/`): `DFTUtils` (FFT/DFT distance-bound math shared by MSIndex/MASS/postFilter),
`rtreemulti/` (a vendored multi-dimensional R-tree library — read this before assuming an off-the-shelf R-tree
package is in use), `CandidateSubsequence`/`CandidateMVSubsequence`/`CandidateSegment` (the core result/candidate
value types threaded through every algorithm).

**Data I/O** (`io.github.io/`): `DataLoader` reads a dataset directory into `DataManager.data` (per the format the
individual dataset dirs under `data/` follow); `WorkloadGenerator` builds query workloads either from held-out
series or by adding noise to indexed series (`queryFromIndexed`/`queryNoiseEps`). Dataset-specific prep notebooks
live in `data/*.ipynb` (synthetic generation, UEA prep, normalization, stats) — there's no committed
`data/instructions.md` despite the README pointing to one.

**Params/stats plumbing:** `Parameters` is a single static-field class serving as both CLI-args destination and
run-statistics accumulator (counters like `nBoundComputations`, `msIndexPPEarlyStops`, timings like
`indexBuildTime`/`queryTime`). `printParams()`/`saveParams()` reflectively dump every non-`@Hide` static field, so
new fields need `@Hide` if they're derived/internal and shouldn't show up as a "parameter" in `output/runs.csv`.

## Working in this repo

- Large local data/artifact directories (`data/`, `index_caches/`, `artifacts*/`, `logs/`, `preprocessed.tar`,
  `.venv/`, `target/`) are present on disk but mostly git-ignored working state, not things to inspect wholesale —
  grep for specific filenames rather than listing these directories.
- `msindex`, `msindexpp`, and `msindexshared` packages duplicate a lot of logic (MBR, bounded priority queue,
  dimension weighting, segmentation). When fixing a bug in one, check whether the same bug exists in the others.
- The Python trainers (`train_msindexpp_gbdt.py`, `train_hybrid_gbdt.py`) export models as portable JSON consumed
  by hand-rolled Java inference (`PortableGBDTModel`, `PortableGBDTDecisionModel`) — there's no shared schema
  validation between the two sides, so if you change the Python export format, update the Java parser in the same
  change.
