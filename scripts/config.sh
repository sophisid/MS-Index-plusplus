# =============================================================================
# config.sh — Edit this file before running run_suite.sh
# =============================================================================

# Base directory that contains the UEA subdirectories.  The experiment
# wrappers below intentionally use only SELECTED_DATASETS; this path is only
# needed for DuckDuckGeese.
# e.g. $UEA_BASE_DIR/ArticularyWordRecognition, $UEA_BASE_DIR/Epilepsy, ...
UEA_BASE_DIR="/path/to/Multivariate_msindex"

# Custom datasets
SYNTHETIC_DIR="/path/to/synthetic"
WEATHER_DIR="/path/to/weather"
STOCKS_DIR="/path/to/stocks"

# Wind channel directory WITHOUT pitch_angle (pitch_angle is constant at
# maxM=10000 and causes query workload generation to hang).
WIND_DIR="/path/to/wind_no_pitch"

# Dataset allow-list used by all batch wrappers.  Keep this list small: the
# full UEA suite is useful for the original paper benchmark, but is not part
# of the current rerun and can fill the experiment machine.
SELECTED_DATASETS=(synthetic stocks wind weather DuckDuckGeese)

# Number of queries per experiment combination
N_QUERIES=100

# Noise and recall grids used by the requested reruns.  Individual wrappers
# may override these through their environment.
NOISE_LIST="0.1 0.3 0.5 0.7 0.9"
TARGET_RECALLS="0.80 0.85 0.90 0.95 0.99"

# Parallel combos per dataset. Set to the number of available CPU cores for
# faster runs; 1 = fully sequential.
MAX_WORKERS=1

SEED=1
