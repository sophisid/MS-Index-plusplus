# MS-Index: Fast Subsequence Search for Multivariate Time Series under Euclidean Distance

This project implements the MS-Index algorithm as described in the paper titled "MS-Index: Fast Subsequence Search for Multivariate Time Series under Euclidean Distance".

## Running the Code

To run the code, execute the [run.sh]() script. This script sets various parameters and runs the Java program with the specified configuration.
To change the parameters, modify the [run.sh]() script.

```bash
./run.sh
```

## Parameters
The parameters in the `run.sh` script are as follows:
- `algorithmType`: The type of algorithm to use (MSINDEX, MSINDEXPP, HYBRID, BRUTE_FORCE, MASS, ST_INDEX, DSTREE, default: MSINDEX)
- `dataPath`: Path to the data directory (default: data/synthetic)
- `N`: Number of time series to read from the dataset (default: 100)
- `channels`: Number of channels to read for each time series (-1 means all channels, default: -1)
- `nQueryChannels`: Number of query channels to use (-1 means all channels, default: -1)
- `qLen`: Length of the query (default: 1024)
- `maxM`: Maximum number of data points to consider for each time series (default: 100000)
- `K`: Number of nearest neighbors to find (default: 1)
- `normalize`: Whether to consider normalized subsequences or not (default: false)
- `nQueries`: Number of queries to run (default: 100)
- `runtimeMode`: Runtime mode (FULL, FULL_NO_STORE, INDEX, INDEX_NO_STORE, QUERY, default: FULL_NO_STORE)
- `experimentId`: Experiment identifier (default: 1)
- `seed`: Seed for random number generation (default: 1)
- `parallel`: Whether to run in parallel (default: false)
- `queryFromIndexed`: Whether to generate queries from indexed data (default: false)
- `queryNoiseEps`: Level of noise added to queries when generated from indexed data (default: 0.1)

## Data
Instructions on how to get the data from the paper can be found in [data/instructions.md](data/instructions.md).

## MSIndexPP Early Termination

The `MSINDEXPP` now supports:
- collecting per-query, per-checkpoint training rows during the second pass
- training an external Python GBDT model per dataset
- loading the exported model in Java and deciding whether to stop the second-pass traversal early

Each training row is a checkpoint snapshot of the form:
- `[features_at_checkpoint] -> continue`
- `[features_at_checkpoint] -> stop`

During training-data collection, Java runs the full second pass once, then replays each checkpoint prefix through `postFilter(...)` to measure the achieved top-k overlap against the final exact top-k. The row is labeled `stop` when that overlap is at least `msindexpp.accuracyTarget`; otherwise it is labeled `continue`.

The extra `MSINDEXPP` configuration is passed as JVM system properties:
- `-Dmsindexpp.trainingEnabled=true`
- `-Dmsindexpp.earlyTerminationEnabled=true`
- `-Dmsindexpp.checkpointInterval=64`
- `-Dmsindexpp.accuracyTarget=0.9`
- `-Dmsindexpp.modelScoreThreshold=0.5`
- `-Dmsindexpp.artifactsDir=artifacts/msindexpp`
- `-Dmsindexpp.trainingDataPath=...` (optional override)
- `-Dmsindexpp.modelPath=...` (optional override)

Example: collect training data with `MSINDEXPP`

```bash
java \
  -Dmsindexpp.trainingEnabled=true \
  -Dmsindexpp.checkpointInterval=64 \
  -cp target/MS-Index-1.0-jar-with-dependencies.jar \
  io/github/Main \
  MSINDEXPP data/synthetic 100 100000 -1 -1 730 1 false 100 FULL_NO_STORE 1 1 false true 0.1
```

This writes dataset-specific JSONL training rows under `artifacts/msindexpp/<dataset>/training-data.jsonl` by default.

Train the Python GBDT model:

```bash
python3 scripts/train_msindexpp_gbdt.py \
  --input artifacts/msindexpp/synthetic/training-data.jsonl \
  --output artifacts/msindexpp/synthetic/gbdt-model.json
```

The trainer now:
- splits train/test by `query_id` when possible, so checkpoints from the same query do not leak into both sets
- applies inverse-frequency class weights, which helps when later checkpoints produce many more `stop` rows than `continue` rows

Then enable inference at query time:

```bash
java \
  -Dmsindexpp.earlyTerminationEnabled=true \
  -Dmsindexpp.checkpointInterval=64 \
  -Dmsindexpp.modelScoreThreshold=0.5 \
  -cp target/MS-Index-1.0-jar-with-dependencies.jar \
  io/github/Main \
  MSINDEXPP data/synthetic 100 100000 -1 -1 730 1 false 100 FULL_NO_STORE 1 1 false true 0.1
```

The Python trainer currently expects `numpy` and `scikit-learn` to be installed in the environment.

For a one-shot train + evaluate run, use:

```bash
bash scripts/run_msindexpp_train_eval.sh
```

Useful environment overrides for the script include `TRAIN_QUERIES`, `EVAL_QUERIES`, `CHECKPOINT_INTERVAL`, `MODEL_SCORE_THRESHOLD`, `DATA_PATH`, `N`, `Q_LEN`, and `K`.

## Hybrid Start-Time Routing

The `HYBRID` algorithm builds both `MASS` and `MSINDEX`, extracts a fixed query-level feature vector before search, and predicts which exact method should answer the query.

The `HYBRID` routing model supports:
- collecting per-query training rows to learn `MASS` vs `MSINDEX`
- loading a portable Python GBDT model in Java
- evaluating predicted-vs-actual best algorithm per query

Useful JVM properties:
- `-Dhybrid.trainingCsv=...` to collect training rows
- `-Dhybrid.evaluationCsv=...` to log evaluation rows
- `-Dhybrid.modelPath=...` to load a trained routing model
- `-Dhybrid.artifactsDir=artifacts/hybrid` to change the default artifact root

Train a routing model:

```bash
bash scripts/HYBRID_jobs.sh --submit
```

Run one dataset directly with `bash scripts/HYBRID_jobs.sh --dataset synthetic`.
