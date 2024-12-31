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

# Window sizes for rolling statistics
window_sizes = [5]


def generate_feature_columns():
    """
    Generates a list of feature columns for the model.
    The feature columns include 'speed', 'acceleration', 'ext_temp',
    and rolling statistics (mean, std) over time windows defined in window_sizes.
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
    and returns the processed data.

    Parameters:
        file (str): Path to the CSV file.
        trip_id (int): An identifier to differentiate each trip.

    Returns:
        pd.DataFrame or None:
            Processed DataFrame with rolling features and a 'trip_id'.
            Returns None if file reading fails or required columns are missing.
    """
    try:
        data = pd.read_csv(file)

        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

            # Calculate rolling features
            for w in window_sizes:
                time_window = w * 2
                data[f'mean_accel_{time_window}'] = (
                    data['acceleration'].rolling(window=w, min_periods=1).mean().bfill()
                )
                data[f'std_accel_{time_window}'] = (
                    data['acceleration']
                    .rolling(window=w, min_periods=1).std().bfill().fillna(0)
                )
                data[f'mean_speed_{time_window}'] = (
                    data['speed'].rolling(window=w, min_periods=1).mean().bfill()
                )
                data[f'std_speed_{time_window}'] = (
                    data['speed']
                    .rolling(window=w, min_periods=1).std().bfill().fillna(0)
                )

            data['trip_id'] = trip_id
            return data
    except Exception as e:
        print(f"Error processing file {file}: {e}")

    return None


def scale_data(df, scaler=None):
    """
    Applies MinMax scaling to FEATURE_COLS. If a scaler is not provided,
    one is created based on predefined minimum/maximum values.

    Parameters:
        df (pd.DataFrame): The DataFrame to scale.
        scaler (MinMaxScaler, optional):
            If None, a new MinMaxScaler is created and fit.
            If provided, it is used to transform the data.

    Returns:
        (pd.DataFrame, MinMaxScaler):
            A tuple of the scaled DataFrame and the scaler used.
    """
    if scaler is None:
        # min_vals / max_vals 설정
        min_vals = [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN]
        max_vals = [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX]

        # window_sizes에 맞춘 rolling feature의 min/max 가정값
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
    Integrates the predicted power (Power_ml) and actual measured power (Power_data)
    over the trip duration using the trapezoidal rule.

    Parameters:
        trip_data (pd.DataFrame): DataFrame containing trip data.

    Returns:
        (float, float):
            (ml_integral, data_integral)
            - ml_integral : integral of the predicted power
            - data_integral: integral of the actual power
    """
    trip_data = trip_data.sort_values(by='time')
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    trip_data['Power_ml'] = trip_data['y_pred']

    ml_cum_integral = cumulative_trapezoid(trip_data['Power_ml'].values, time_seconds, initial=0)
    ml_integral = ml_cum_integral[-1]

    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return ml_integral, data_integral


# ----------------------------
# Optuna Objective Function
# ----------------------------
def cv_objective(trial, train_files, scaler):
    """
    Objective function for Optuna, performing K-Fold CV on a set of training files.

    Parameters:
        trial (optuna.trial.Trial): The current trial object.
        train_files (list): List of CSV file paths for training.
        scaler (MinMaxScaler): Scaler for feature normalization.

    Returns:
        float: The mean RMSE across all folds.
    """
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

        # Process training files
        fold_train_data_list = []
        for i, f in enumerate(fold_train_files):
            d = process_single_file(f, trip_id=i)
            if d is not None:
                fold_train_data_list.append(d)
        fold_train_data = pd.concat(fold_train_data_list, ignore_index=True)

        # Process validation files
        fold_val_data_list = []
        for j, f in enumerate(fold_val_files):
            d = process_single_file(f, trip_id=1000 + j)
            if d is not None:
                fold_val_data_list.append(d)
        fold_val_data = pd.concat(fold_val_data_list, ignore_index=True)

        # Scale
        fold_train_data_scaled, _ = scale_data(fold_train_data.copy(), scaler)
        fold_val_data_scaled, _ = scale_data(fold_val_data.copy(), scaler)

        X_tr = fold_train_data_scaled[FEATURE_COLS]
        y_tr = fold_train_data_scaled['Power_data']
        X_val = fold_val_data_scaled[FEATURE_COLS]
        y_val = fold_val_data_scaled['Power_data']

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


def tune_hyperparameters(train_files, scaler, plot):
    """
    Tunes hyperparameters using Optuna on the training files.
    Conducts K-Fold cross-validation and optimizes for the lowest mean RMSE.
    Optionally plots the trial history.

    Parameters:
        train_files (list): List of training file paths.
        scaler (MinMaxScaler): The feature scaler.
        plot (bool): If True, generate and show a plot of the trial outcomes.

    Returns:
        dict: The best hyperparameters found by Optuna.
    """
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: cv_objective(trial, train_files, scaler), n_trials=100)

    print(f"Best trial: {study.best_trial.params}")

    if plot:
        # Convert all trial results to a DataFrame
        trials_df = study.trials_dataframe()
        # Export to CSV
        trials_df.to_csv("optuna_trials_results.csv", index=False)
        print("All trial results have been saved to optuna_trials_results.csv")

        # Filter only completed trials
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        trial_numbers = [t.number for t in complete_trials]
        trial_values = [t.value for t in complete_trials]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, trial_values, marker='o', linestyle='-', label='Trials')

        # Highlight the best trial
        best_trial = study.best_trial
        best_trial_number = best_trial.number
        best_trial_value = best_trial.value
        plt.plot(best_trial_number, best_trial_value, marker='o', markersize=12, color='red', label='Best Trial')

        # Set axes and title
        plt.xlabel('Trial')
        plt.ylabel('CVE (Mean of Fold RMSE)')
        plt.title('CVE per Trial during Bayesian Optimization')
        plt.legend()

        # Display best parameters on the plot
        best_params_str = '\n'.join([f"{k}: {v:.4f}" for k, v in best_trial.params.items()])
        plt.text(
            0.95, 0.95, best_params_str, transform=plt.gca().transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        try:
            plt.savefig("Figure_Optuna_CVE.png", dpi=300, bbox_inches='tight')
            print("Figure saved: Figure_Optuna_CVE.png")
        except Exception as e:
            print(f"Error saving figure: {e}")

        # Show the plot
        plt.show()

    return study.best_trial.params


def train_final_model(X_train, y_train, best_params):
    """
    Trains the final XGBoost model using the best hyperparameters.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values (Power_data).
        best_params (dict): Best hyperparameters (e.g., from Optuna).

    Returns:
        xgb.Booster: A trained XGBoost Booster object.
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
    Main workflow function that:
    1. Splits files into train/test sets
    2. Processes and scales the data
    3. Tunes hyperparameters (unless predefined are provided)
    4. Trains the final XGBoost model
    5. Evaluates on test data

    Parameters:
        files (list): List of CSV file paths to process.
        plot (bool): Whether to plot Optuna trial results.
        save_dir (str or None): Directory to save model/scaler. If None, skip saving.
        predefined_best_params (dict, optional):
            If provided, skip tuning and directly use these parameters.

    Returns:
        (list, MinMaxScaler):
            - A list of result dictionaries (e.g., RMSE, MAPE, best_params).
            - The scaler used for feature transformation.
    """
    if not files:
        print("No files provided.")
        return

    # Split files into train/test (80:20)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Process training data
    train_data_list = []
    for i, f in enumerate(train_files):
        d = process_single_file(f, trip_id=i)
        if d is not None:
            train_data_list.append(d)
    train_data = pd.concat(train_data_list, ignore_index=True)

    # Fit scaler on train data
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

    # Prepare train/test sets
    X_train = train_data_scaled[FEATURE_COLS]
    y_train = train_data_scaled['Power_data']
    X_test = test_data_scaled[FEATURE_COLS]
    y_test = test_data_scaled['Power_data']

    # Hyperparameter tuning (if needed)
    if predefined_best_params is None:
        best_params = tune_hyperparameters(train_files, scaler, plot)
    else:
        best_params = predefined_best_params
        print(f"Using predefined best_params: {best_params}")

    # Train final model
    bst = train_final_model(X_train, y_train, best_params)

    # Test performance
    y_pred_test = bst.predict(xgb.DMatrix(X_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"Test RMSE with best_params: {test_rmse:.4f}")

    test_data_scaled['y_pred'] = y_pred_test

    # Compute integrals on test set (trip by trip)
    test_trip_groups = test_data_scaled.groupby('trip_id')
    ml_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        ml_integral, data_integral = integrate_and_compare(group)
        ml_integrals_test.append(ml_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test), np.array(ml_integrals_test))
    print("Test Set Integration Metrics:")
    print(f"MAPE: {mape_test:.2f}%")

    results = [{
        'rmse': test_rmse,
        'test_mape': mape_test,
        'best_params': best_params
    }]

    # Optionally save the model and scaler
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
