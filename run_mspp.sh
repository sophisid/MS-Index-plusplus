#!/bin/bash

# Base parameters (σταθερά σε όλα τα runs)
algorithmType=MSINDEXPP         # algorithm type: MSINDEX, MSINDEXPP, BRUTE_FORCE, MASS, ST_INDEX, DSTREE, KV_MATCH
dataPath=data/synthetic         # path to the data directory

N=100                           # <<< ΣΤΑΘΕΡΟ N=100 >>>
channels=-1                     # -1 means all channels
nQueryChannels=-1               # -1 means all channels
qLen=730                        # query length
maxM=100000                     # maximum length of the time series

normalize_vals=("false" "true") # normalized or not
nQueries_vals=(100 1000)        # <<< ΑΥΤΟ ΘΑ ΚΑΝΕΙ SWEEP >>>
K_vals=(1 5)                    # K = 1, 5
noise_vals=(0.1 0.3 0.5 0.9)    # noise levels

runtimeMode=FULL_NO_STORE       # FULL, FULL_NO_STORE, INDEX, INDEX_NO_STORE, QUERY
seed=5                          # random seed
parallel=false                  # run in multi-threaded mode or not
queryFromIndexed=true           # query from indexed subsequences or not

echo "Building project with Maven..."
mvn -q -DskipTests clean package || { echo "Maven build failed"; exit 1; }

JAR=target/MS-Index-1.0-jar-with-dependencies.jar

if [ ! -f "$JAR" ]; then
  echo "Jar not found at $JAR"
  exit 1
fi

echo "Starting grid of experiments..."

expId=0

for normalize in "${normalize_vals[@]}"; do
  for nQueries in "${nQueries_vals[@]}"; do
    for K in "${K_vals[@]}"; do
      for noise in "${noise_vals[@]}"; do

        expId=$((expId + 1))

        echo "============================================================"
        echo "Run #$expId:"
        echo "  algorithmType = $algorithmType"
        echo "  dataPath      = $dataPath"
        echo "  N             = $N"
        echo "  nQueries      = $nQueries"
        echo "  K             = $K"
        echo "  normalize     = $normalize"
        echo "  noise         = $noise"
        echo "  experimentId  = $expId"
        echo "============================================================"

        java \
          -cp "$JAR" \
          io.github.Main \
          "$algorithmType" \
          "$dataPath" \
          "$N" \
          "$maxM" \
          "$channels" \
          "$nQueryChannels" \
          "$qLen" \
          "$K" \
          "$normalize" \
          "$nQueries" \
          "$runtimeMode" \
          "$expId" \
          "$seed" \
          "$parallel" \
          "$queryFromIndexed" \
          "$noise"

      done
    done
  done
done

echo "All experiments finished. Total runs: $expId"
