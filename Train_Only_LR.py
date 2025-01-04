import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Functions import calculate_rmse, calculate_mape
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import cumulative_trapezoid

# This code contains comments generated using ChatGPT.

def process_single_file(file):
    """
    Process a single CSV file and select relevant columns for modeling.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_data' in data.columns:
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    Process multiple CSV files in parallel, compute rolling statistics,
    and apply feature scaling.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6
    ACCELERATION_MIN = -15
    ACCELERATION_MAX = 9
    TEMP_MIN = -30
    TEMP_MAX = 50

    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    df_list = []
    with ProcessPoolExecutor() as executor:
        future_to_info = {
            executor.submit(process_single_file, file): (idx, file)
            for idx, file in enumerate(files)
        }

        for future in as_completed(future_to_info):
            idx, file = future_to_info[future]
            try:
                data = future.result()
                if data is not None:
                    # Convert 'time' to datetime
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # Assign a trip_id based on the file index
                    data['trip_id'] = idx

                    # Calculate rolling statistics with a window of 5
                    data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
                    data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
                    data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    # Concatenate all DataFrames
    full_data = pd.concat(df_list, ignore_index=True)

    # Initialize the scaler if none is provided
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # Apply scaling to the feature columns only
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler


def integrate_and_compare(trip_data):
    """
    Integrate 'Power_data' (actual) and the model prediction ('y_pred') over time.
    Returns (predicted_integral, actual_integral).
    """
    # Sort by time to ensure correct chronological order
    trip_data = trip_data.sort_values(by='time')

    # Convert timestamps to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Predicted power comes directly from model predictions
    predicted_power = trip_data['y_pred']
    predicted_cum_integral = cumulative_trapezoid(predicted_power.values, time_seconds, initial=0)
    predicted_integral = predicted_cum_integral[-1]

    actual_power = trip_data['Power_data']
    actual_cum_integral = cumulative_trapezoid(actual_power.values, time_seconds, initial=0)
    actual_integral = actual_cum_integral[-1]

    return predicted_integral, actual_integral


def train_model_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model using the given training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_validate_test(vehicle_files):
    """
    1) Split 20% of the files for testing and 80% for training.
    2) Perform 5-Fold cross-validation on the training portion (Linear Regression).
    3) Among the 5-Fold results, select the fold whose Validation RMSE is closest
       to the median as the best model.
    4) Evaluate the selected best model on the test set (MAPE, RMSE).
    """
    if not vehicle_files:
        raise ValueError("No files provided. The 'vehicle_files' list is empty.")

    # Step (1): Split into train and test files
    all_files = vehicle_files
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    if len(train_files) == 0:
        raise ValueError("No training files were found after splitting. Check file list or split ratio.")
    if len(test_files) == 0:
        raise ValueError("No test files were found after splitting. Check file list or split ratio.")

    # Step (2): 5-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_files = np.array(train_files)

    fold_results = []
    fold_models = []
    fold_scalers = []

    feature_cols = [
        'speed', 'acceleration', 'ext_temp',
        'mean_accel_10', 'std_accel_10',
        'mean_speed_10', 'std_speed_10'
    ]

    for fold_num, (fold_train_idx, fold_val_idx) in enumerate(kf.split(train_files), start=1):
        fold_train_files = train_files[fold_train_idx]
        fold_val_files = train_files[fold_val_idx]

        # Prepare training/validation data
        train_data, scaler = process_files(fold_train_files)
        val_data, _ = process_files(fold_val_files, scaler=scaler)

        # Features and label
        X_train = train_data[feature_cols]
        y_train = train_data['Power_data']

        X_val = val_data[feature_cols]
        y_val = val_data['Power_data']

        # Train the model
        model = train_model_linear_regression(X_train, y_train)

        # Validation predictions
        val_data['y_pred'] = model.predict(X_val)

        # Integrate over time by trip_id
        val_trip_groups = val_data.groupby('trip_id')
        predicted_integrals_val, data_integrals_val = [], []

        for _, group in val_trip_groups:
            predicted_integral, data_integral = integrate_and_compare(group)
            predicted_integrals_val.append(predicted_integral)
            data_integrals_val.append(data_integral)

        # Calculate MAPE using the integrated values
        mape_val = calculate_mape(np.array(data_integrals_val),
                                  np.array(predicted_integrals_val))

        # Calculate RMSE using the time-series data directly
        rmse_val = calculate_rmse(y_val, val_data['y_pred'])

        fold_results.append({
            'fold': fold_num,
            'rmse': rmse_val,
            'mape': mape_val
        })
        fold_models.append(model)
        fold_scalers.append(scaler)

        print(f"[Fold {fold_num}] Validation RMSE = {rmse_val:.4f}, MAPE = {mape_val:.2f}%")

    # Step (3): Select the fold whose RMSE is closest to the median
    val_rmse_values = [res['rmse'] for res in fold_results]
    median_rmse = np.median(val_rmse_values)
    closest_index = np.argmin(np.abs(np.array(val_rmse_values) - median_rmse))

    best_model_info = fold_results[closest_index]
    best_model = fold_models[closest_index]
    best_scaler = fold_scalers[closest_index]
    best_fold = best_model_info['fold']

    print(f"\n[Best Model Selection]")
    print(f"  => Fold {best_fold} selected.")
    print(f"     (Validation RMSE: {best_model_info['rmse']:.4f}, MAPE: {best_model_info['mape']:.2f}%)")

    # Step (4): Evaluate the best model on the test set
    test_data, _ = process_files(test_files, scaler=best_scaler)

    X_test = test_data[feature_cols]
    y_test = test_data['Power_data']

    test_data['y_pred'] = best_model.predict(X_test)

    test_trip_groups = test_data.groupby('trip_id')
    predicted_integrals_test, data_integrals_test = [], []

    for _, group in test_trip_groups:
        predicted_integral, data_integral = integrate_and_compare(group)
        predicted_integrals_test.append(predicted_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test),
                               np.array(predicted_integrals_test))
    rmse_test = calculate_rmse(y_test, test_data['y_pred'])

    print(f"\n[Test Set Evaluation using Best Model (Fold {best_fold})]")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.2f}%")
    print("------------------------------------")

    results = [{
        'fold_results': fold_results,
        'best_fold': best_fold,
        'best_model': best_model,
        'test_rmse': rmse_test,
        'test_mape': mape_test
    }]

    return results, best_scaler
