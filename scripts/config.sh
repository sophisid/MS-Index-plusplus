# =============================================================================
# config.sh — Edit this file before running run_suite.sh
# =============================================================================

# Base directory that contains the 30 UEA subdirectories
# e.g. $UEA_BASE_DIR/ArticularyWordRecognition, $UEA_BASE_DIR/Epilepsy, ...
UEA_BASE_DIR="/path/to/Multivariate_msindex"

# Custom datasets
SYNTHETIC_DIR="/path/to/synthetic"
WEATHER_DIR="/path/to/weather"
STOCKS_DIR="/path/to/stocks"

# Wind channel directory WITHOUT pitch_angle (pitch_angle is constant at
# maxM=10000 and causes query workload generation to hang).
WIND_DIR="/path/to/wind_no_pitch"

# Number of queries per experiment combination
N_QUERIES=100

# Parallel combos per dataset. Set to the number of available CPU cores for
# faster runs; 1 = fully sequential.
MAX_WORKERS=1

SEED=1
