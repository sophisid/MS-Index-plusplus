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

## Early termination and hybrid search

`MSINDEXPP` can collect checkpoint training rows, train a gradient boosted decision model, and use that model to stop its second pass early. Enable the stages with JVM properties such as `-Dmsindexpp.trainingEnabled=true`, `-Dmsindexpp.earlyTerminationEnabled=true`, and `-Dmsindexpp.evaluationEnabled=true`. The checkpoint interval, accuracy target, model score threshold, artifact directory, training data path, and model path can be set with the corresponding `msindexpp.*` properties in `Parameters.java`.

Train a model from the collected JSONL rows with `python3 scripts/train_msindexpp_gbdt.py --input <training-data.jsonl> --output <gbdt-model.json>`. `scripts/calibrate_conformal.py` can add a calibrated stopping threshold to the model. The Python trainers require NumPy and scikit-learn.

For a local dataset suite, run `python3 scripts/run_msindexpp_dataset_suite.py --dataset-spec 'name:path:qLen[:defaultN[:maxLen]]'`. The suite uses the current Python interpreter for training; set `TRAINER_PYTHON` to use another one. Run the script with `--help` for dataset, training, evaluation, and benchmark options.

`HYBRID` builds MASS and MSINDEX, then uses query features to route each query. Set `-Dhybrid.trainingCsv=<path>` to collect routing rows, train with `python3 scripts/train_hybrid_gbdt.py --help`, and load the exported model with `-Dhybrid.modelPath=<path>`.
