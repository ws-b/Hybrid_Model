# 1. Data and Vehicle Settings
# ----------------------------------
# Path to the directory containing the dataset
DATA_PATH = "/home/ubuntu/SamsungSTF/Processed_Data/Trips"

# Select the vehicle to be used in the experiment
SELECTED_VEHICLE = "EV6"


# 2. Sampling and Experiment Settings
# ----------------------------------
# List of sample sizes (number of trips) to be evaluated
SAMPLING_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 20000]
# SAMPLING_SIZES = [10] // FOR DEBUGGING

# Number of random sampling iterations for each sample size.
# Smaller sizes have more iterations to ensure statistical significance.
SAMPLING_ITERATIONS = {
    10: 200,
    20: 10,
    50: 5,
    100: 5
}

# 3. Model and Tuning Settings
# ----------------------------------
# List of models to be included in the experiment.
# Available models: "XGBoost", "RandomForest", "MLP", "Transformer", "LinearRegression"
MODELS_TO_RUN = ["XGBoost", "RandomForest", "MLP", "Transformer", "LinearRegression"]

# List of models to perform hyperparameter tuning on.
# If a model's parameters are already tuned, you can remove it from this list.
MODELS_TO_TUNE = ["XGBoost", "RandomForest", "MLP", "Transformer", "LinearRegression"]

# Optuna settings for hyperparameter tuning
N_TRIALS_OPTUNA = 25  # Number of trials for each model's tuning process
# N_TRIALS_OPTUNA = 3 // FOR DEBUGGING

# 4. Feature Settings
# ----------------------------------
# Features to be used for the Transformer model
TRANSFORMER_FEATURES = ['speed', 'acceleration', 'ext_temp']

# 5. 실험에서 제외할 모델 목록
# ----------------------------------
# 여기에 적힌 모델은 튜닝 및 평가 과정에서 건너뜁니다.
# 예: ["RandomForest", "MLP"]
MODELS_TO_SKIP = []
