import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
from Functions import calculate_mape
from scipy.integrate import cumulative_trapezoid
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from optuna.trial import TrialState

# This code contains comments generated using ChatGPT.
# ----------------------------
# Global Variables / Constants
# ----------------------------
SPEED_MIN = 0 / 3.6
SPEED_MAX = 230 / 3.6
ACCELERATION_MIN = -15
ACCELERATION_MAX = 9
TEMP_MIN = -30
TEMP_MAX = 50
ACCEL_STD_MAX = 10
SPEED_STD_MAX = 30

# Window sizes for rolling features
window_sizes = [5]


def generate_feature_columns():
    """
    Creates a list of feature columns for the model, which includes:
    - speed, acceleration, ext_temp
    - Rolling mean and standard deviation for the specified window sizes.
    """
    feature_cols = ['speed', 'acceleration', 'ext_temp']
    for w in window_sizes:
        time_window = w * 2
        feature_cols.extend([
            f'mean_accel_{time_window}',
            f'std_accel_{time_window}',
            f'mean_speed_{time_window}',
            f'std_speed_{time_window}'
        ])
    return feature_cols


FEATURE_COLS = generate_feature_columns()


# ----------------------------
# File Processing Functions
# ----------------------------
def process_single_file(file, trip_id):
    """
    Reads a single CSV file (assumed to be one trip), calculates rolling features,
    and returns the processed DataFrame. Residual = Power_data - Power_phys.

    Parameters:
        file (str): The path to the CSV file.
        trip_id (int): An identifier for the trip (used to differentiate trips).

    Returns:
        pd.DataFrame or None:
            The processed DataFrame containing rolling features and 'Residual',
            or None if the file doesn't contain 'Power_phys' and 'Power_data'.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # Calculate rolling features
            for w in window_sizes:
                time_window = w * 2
                data[f'mean_accel_{time_window}'] = data['acceleration'].rolling(
                    window=w, min_periods=1
                ).mean().bfill()
                data[f'std_accel_{time_window}'] = data['acceleration'].rolling(
                    window=w, min_periods=1
                ).std().bfill().fillna(0)

                data[f'mean_speed_{time_window}'] = data['speed'].rolling(
                    window=w, min_periods=1
                ).mean().bfill()
                data[f'std_speed_{time_window}'] = data['speed'].rolling(
                    window=w, min_periods=1
                ).std().bfill().fillna(0)

            data['trip_id'] = trip_id
            return data
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def scale_data(df, scaler=None):
    """
    Applies MinMax scaling to the columns in FEATURE_COLS.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the features to scale.
        scaler (MinMaxScaler, optional): If None, a new scaler is created based on
            predefined min/max values. Otherwise, the provided scaler is used.

    Returns:
        (pd.DataFrame, MinMaxScaler): The scaled DataFrame and the scaler used.
    """
    if scaler is None:
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]

        # Rolling feature 가정 범위 설정
        window_val_min = [ACCELERATION_MIN, 0, SPEED_MIN, 0]
        window_val_max = [ACCELERATION_MAX, ACCEL_STD_MAX, SPEED_MAX, SPEED_STD_MAX]

        for w in window_sizes:
            min_vals.extend(window_val_min)
            max_vals.extend(window_val_max)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([min_vals, max_vals], columns=FEATURE_COLS))

    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df, scaler


def integrate_and_compare(trip_data):
    """
    Integrates the Hybrid (Power_phys + y_pred) and Power_data over time
    using the trapezoidal rule to compare total energy consumption.

    Parameters:
        trip_data (pd.DataFrame): Data for a single trip.

    Returns:
        (float, float): A tuple (hybrid_integral, data_integral) representing
                        the total energy from the hybrid model and the actual data.
    """
    trip_data = trip_data.sort_values(by='time')
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Hybrid = Power_phys + Residual Prediction
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1]

    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return hybrid_integral, data_integral


# ----------------------------
# Optuna K-Fold CV Objective
# ----------------------------
def cv_objective(trial, train_files, scaler):
    """
    Objective function for Optuna, performing K-Fold cross-validation on the
    training files. Trains an XGBoost model to predict the 'Residual'
    (Power_data - Power_phys).

    Parameters:
        trial (optuna.trial.Trial): The current Optuna trial object.
        train_files (list): List of file paths used for training.
        scaler (MinMaxScaler): The scaler for feature normalization.

    Returns:
        float: The mean RMSE across all folds (the metric to minimize).
    """
    # Hyperparameter sampling
    reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1e5, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-6, 1e5, log=True)
    eta = trial.suggest_float('eta', 0.01, 0.3, log=True)

    params = {
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'eta': eta,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    train_files = np.array(train_files)

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(train_files)):
        fold_train_files = train_files[train_idx]
        fold_val_files = train_files[val_idx]

        # Process training data
        fold_train_data_list = []
        for i, f in enumerate(fold_train_files):
            d = process_single_file(f, trip_id=i)
            if d is not None:
                fold_train_data_list.append(d)
        fold_train_data = pd.concat(fold_train_data_list, ignore_index=True)

        # Process validation data
        fold_val_data_list = []
        for j, f in enumerate(fold_val_files):
            d = process_single_file(f, trip_id=1000 + j)
            if d is not None:
                fold_val_data_list.append(d)
        fold_val_data = pd.concat(fold_val_data_list, ignore_index=True)

        # Scale
        fold_train_data_scaled, _ = scale_data(fold_train_data.copy(), scaler)
        fold_val_data_scaled, _ = scale_data(fold_val_data.copy(), scaler)

        # Train/Val sets
        X_tr = fold_train_data_scaled[FEATURE_COLS]
        y_tr = fold_train_data_scaled['Residual']
        X_val = fold_val_data_scaled[FEATURE_COLS]
        y_val = fold_val_data_scaled['Residual']

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=15,
            verbose_eval=False
        )

        preds = bst.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_list.append(rmse)

    cve = np.mean(rmse_list)
    return cve


def tune_hyperparameters(train_files, scaler, plot=False):
    """
    Performs hyperparameter tuning using Optuna's TPE (or other sampler),
    running cv_objective for K-Fold cross-validation.

    Parameters:
        train_files (list): Paths to training files.
        scaler (MinMaxScaler): Scaler used for feature normalization.
        plot (bool): Whether to visualize and save the trial results.

    Returns:
        dict: Dictionary of the best parameters found by Optuna.
    """
    # Create Optuna study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: cv_objective(trial, train_files, scaler), n_trials=100)

    print(f"Best trial: {study.best_trial.params}")

    if plot:
        # Convert all trial results to DataFrame
        trials_df = study.trials_dataframe()

        # Save as CSV
        trials_df.to_csv("optuna_trials_results.csv", index=False)
        print("All trial results have been saved to optuna_trials_results.csv")

        # Filter only completed trials
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        trial_numbers = [t.number for t in complete_trials]
        trial_values = [t.value for t in complete_trials]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, trial_values, marker='o', linestyle='-', label='Trials')

        # Highlight best trial
        best_trial = study.best_trial
        best_trial_number = best_trial.number
        best_trial_value = best_trial.value

        plt.plot(best_trial_number, best_trial_value, marker='o', markersize=12, color='red', label='Best Trial')

        # Add labels/title
        plt.xlabel('Trial')
        plt.ylabel('CVE (Mean of Fold RMSE)')
        plt.title('CVE per Trial during Bayesian Optimization')

        plt.legend()

        # Annotate best parameters
        best_params_str = '\n'.join([f"{k}: {v:.4f}" for k, v in best_trial.params.items()])
        plt.text(
            0.95, 0.95, best_params_str, transform=plt.gca().transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

        plt.tight_layout()

        # Save figure
        try:
            plt.savefig("Figure_Optuna_CVE.png", dpi=300, bbox_inches='tight')
            print("Figure saved: Figure_Optuna_CVE.png")
        except Exception as e:
            print(f"Error saving figure: {e}")

        # Show plot
        plt.show()

    return study.best_trial.params


def train_final_model(X_train, y_train, best_params):
    """
    Trains the final XGBoost model using the best hyperparameters.

    Parameters:
        X_train (pd.DataFrame): Training features for residual prediction.
        y_train (pd.Series): The residual values (Power_data - Power_phys).
        best_params (dict): Optimal hyperparameters found by Optuna (or predefined).

    Returns:
        xgb.Booster: A trained XGBoost booster object.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'eta': best_params['eta'],
        'reg_lambda': best_params['reg_lambda'],
        'reg_alpha': best_params['reg_alpha'],
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }
    bst = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=False)
    return bst


# ----------------------------
# Main Workflow
# ----------------------------
def run_workflow(files, plot=False, save_dir=None, predefined_best_params=None):
    """
    Main workflow for training a Hybrid XGBoost model to predict the residual (Power_data - Power_phys):
    1) Splits data into train/test based on file-level (80:20).
    2) Processes and scales the data.
    3) Conducts hyperparameter tuning (Optuna) unless predefined parameters are provided.
    4) Trains the final model with the best parameters.
    5) Evaluates on test data by RMSE and integral-based MAPE.

    Parameters:
        files (list): A list of CSV file paths to use for training/testing.
        plot (bool): If True, plots the Optuna trial results.
        save_dir (str): Directory path to save the trained model and scaler. Defaults to "models".
        predefined_best_params (dict, optional): If provided, skip Optuna tuning and use these parameters.

    Returns:
        (list, MinMaxScaler): A tuple containing:
            - A list of result dictionaries (RMSE, MAPE, best_params).
            - The scaler used for data transformation.
    """
    if not files:
        print("No files provided.")
        return

    # Split files into train/test (80:20)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Process train data
    train_data_list = []
    for i, f in enumerate(train_files):
        d = process_single_file(f, trip_id=i)
        if d is not None:
            train_data_list.append(d)
    train_data = pd.concat(train_data_list, ignore_index=True)

    # Scale train data
    train_data_scaled, scaler = scale_data(train_data.copy(), scaler=None)

    # Process test data
    test_data_list = []
    for j, f in enumerate(test_files):
        d = process_single_file(f, trip_id=1000 + j)
        if d is not None:
            test_data_list.append(d)
    test_data = pd.concat(test_data_list, ignore_index=True)

    # Scale test data
    test_data_scaled, _ = scale_data(test_data.copy(), scaler)

    # Prepare features/labels
    X_train = train_data_scaled[FEATURE_COLS]
    y_train = train_data_scaled['Residual']

    X_test = test_data_scaled[FEATURE_COLS]
    y_test = test_data_scaled['Residual']

    # Either tune hyperparameters or use predefined
    if predefined_best_params is None:
        best_params = tune_hyperparameters(train_files, scaler, plot=plot)
    else:
        best_params = predefined_best_params
        print(f"Using predefined best_params: {best_params}")

    # Train final model
    bst = train_final_model(X_train, y_train, best_params)

    # Evaluate on test data
    y_pred_test = bst.predict(xgb.DMatrix(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Test RMSE with best_params: {test_rmse:.4f}")

    test_data_scaled['y_pred'] = y_pred_test

    # Calculate integrals (trip by trip)
    test_trip_groups = test_data_scaled.groupby('trip_id')
    hybrid_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        hybrid_integral, data_integral = integrate_and_compare(group)
        hybrid_integrals_test.append(hybrid_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test), np.array(hybrid_integrals_test))
    print(f"Test Set Integration Metrics:")
    print(f"MAPE: {mape_test:.2f}%")

    results = [{
        'rmse': test_rmse,
        'test_mape': mape_test,
        'best_params': best_params
    }]

    # (Optional) Save the model and scaler
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_file = os.path.join(save_dir, "XGB_best_model.model")
        bst.save_model(model_file)
        print(f"Best model saved with Test RMSE: {test_rmse:.4f}")

        scaler_path = os.path.join(save_dir, 'XGB_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at {scaler_path}")

    return results, scaler
