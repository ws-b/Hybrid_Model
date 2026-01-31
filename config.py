# 1. Data and Vehicle Settings
# ----------------------------------
# Path to the directory containing the dataset
DATA_PATH = "sample_trips"

# Select the vehicle to be used in the experiment
SELECTED_VEHICLE = "EV6"

# Path to save results
RESULTS_DIR = "results"


# 2. Sampling and Experiment Settings
# ----------------------------------
# ----------------------------------
# DEBUG MODE
# ----------------------------------
# Set to True to run a fast verification test (small data, few iterations).
# Set to False to run the full experiment.
DEBUG_MODE = False


# 2. Sampling and Experiment Settings
# ----------------------------------
if DEBUG_MODE:
    # Minimal settings for fast debugging
    SAMPLING_SIZES = [10]
    SAMPLING_ITERATIONS = {10: 1}
    N_TRIALS_OPTUNA = 2
else:
    # Full experiment settings
    SAMPLING_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000]
    
    # Number of random sampling iterations for each sample size.
    SAMPLING_ITERATIONS = {
        10: 200,
        20: 10,
        50: 5,
        100: 5
    }

    # Optuna settings for hyperparameter tuning
    N_TRIALS_OPTUNA = 25  # Number of trials for each model's tuning process

# 3. Model and Tuning Settings
# ----------------------------------
# List of models to be included in the experiment.
# Available models: "XGBoost", "RandomForest", "MLR"
MODELS_TO_RUN = [
    "XGBoost", "RandomForest", "MLR"
    ]

# List of models to perform hyperparameter tuning on.
# If a model's parameters are already tuned, you can remove it from this list.
MODELS_TO_TUNE = [
    "XGBoost", "RandomForest", "MLR"
    ]

# 4. Feature Settings
# ----------------------------------

# 5. 실험에서 제외할 모델 목록
# ----------------------------------
# 여기에 적힌 모델은 튜닝 및 평가 과정에서 건너뜁니다.
# 예: ["RandomForest"]
MODELS_TO_SKIP = []
