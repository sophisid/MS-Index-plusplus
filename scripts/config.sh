# =============================================================================
# config.sh — Edit this file before running run_suite.sh
# =============================================================================

# Base directory that contains the 30 UEA subdirectories
# e.g. $UEA_BASE_DIR/ArticularyWordRecognition, $UEA_BASE_DIR/Epilepsy, ...
UEA_BASE_DIR="/home/jdhondt/MS-Index-plusplus/data"

# Custom datasets
SYNTHETIC_DIR="/home/jdhondt/MS-Index-plusplus/data/synthetic"
WEATHER_DIR="/home/jdhondt/MS-Index-plusplus/data/weather"
STOCKS_DIR="/home/jdhondt/MS-Index-plusplus/data/stocks"

# Wind channel directory WITHOUT pitch_angle (pitch_angle is constant at
# maxM=10000 and causes query workload generation to hang).
WIND_DIR="/home/jdhondt/MS-Index-plusplus/data/wind_no_pitch_data"

# Number of queries per experiment combination
N_QUERIES=100

# Parallel combos per dataset. Set to the number of available CPU cores for
# faster runs; 1 = fully sequential.
MAX_WORKERS=1

SEED=1
