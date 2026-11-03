# MS-Index++: Adaptive Exact and Approximate Top-k Subsequence Search for Multivariate Time Series

This repository accompanies the paper **"MS-Index++: Adaptive Exact and Approximate Top-k Subsequence Search for Multivariate Time Series."** It is an extension of the earlier conference paper, ["MS-Index: Fast Top-k Subsequence Search for Multivariate Time Series under Euclidean Distance"](https://doi.org/10.14778/3773749.3773751) (PVLDB 19(2), 2025).

The earlier paper introduced MS-Index for exact subsequence search with query-time channel selection. MS-Index++ adds a learned selector that chooses between exact MS-Index and MASS for each query, and learned early termination for approximate search at a requested recall target. The new paper also studies query difficulty and evaluates these extensions.

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
- `qLen`: Length of the query (default: 730)
- `maxM`: Maximum number of data points to consider for each time series (default: 100000)
- `K`: Number of nearest neighbors to find (default: 1)
- `normalize`: Whether to consider normalized subsequences or not (default: false)
- `nQueries`: Number of queries to run (default: 100)
- `runtimeMode`: Runtime mode (FULL, FULL_NO_STORE, INDEX, INDEX_NO_STORE, QUERY, default: FULL_NO_STORE)
- `experimentId`: Experiment identifier (default: 1)
- `seed`: Seed for random number generation (default: 1)
- `parallel`: Whether to run in parallel (default: false)
- `queryFromIndexed`: Whether to generate queries from indexed data (default: true)
- `queryNoiseEps`: Level of noise added to queries when generated from indexed data (default: 0.1)

## Data
The time-series datasets can be browsed in the [Time Series Classification dataset catalog](https://www.timeseriesclassification.com/dataset.php). Instructions for obtaining and preparing the datasets used in the paper are in [data/instructions.md](data/instructions.md).

## Early termination and hybrid search

`MSINDEXPP` can collect checkpoint training rows, train a gradient boosted decision model, and use that model to stop its second pass early. Enable the stages with JVM properties such as `-Dmsindexpp.trainingEnabled=true`, `-Dmsindexpp.earlyTerminationEnabled=true`, and `-Dmsindexpp.evaluationEnabled=true`. The checkpoint interval, accuracy target, model score threshold, artifact directory, training data path, and model path can be set with the corresponding `msindexpp.*` properties in `Parameters.java`.

Train a model from the collected JSONL rows with `python3 scripts/train_msindexpp_gbdt.py --input <training-data.jsonl> --output <gbdt-model.json>`. `scripts/calibrate_conformal.py` can add a calibrated stopping threshold to the model. The Python trainers require NumPy and scikit-learn.

For a local dataset suite, run `python3 scripts/run_msindexpp_dataset_suite.py --dataset-spec 'name:path:qLen[:defaultN[:maxLen]]'`. The suite uses the current Python interpreter for training; set `TRAINER_PYTHON` to use another one. Run the script with `--help` for dataset, training, evaluation, and benchmark options.

`HYBRID` builds MASS and MSINDEX, then uses query features to route each query. Set `-Dhybrid.trainingCsv=<path>` to collect routing rows, train with `python3 scripts/train_hybrid_gbdt.py --help`, and load the exported model with `-Dhybrid.modelPath=<path>`.
