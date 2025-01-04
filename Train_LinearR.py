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
    Process a single CSV file to compute residuals and select relevant columns.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            # Calculate the residual by subtracting 'Power_phys' from 'Power_data'
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp',
                         'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    Process multiple CSV files in parallel, calculate rolling statistics,
    and apply feature scaling.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # km/h -> m/s
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
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
                    # Convert the 'time' column to datetime format
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # Assign the trip ID based on the file index
                    data['trip_id'] = idx

                    # Calculate rolling statistics with a window of size 5
                    data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
                    data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
                    data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    # Concatenate all DataFrames into one
    full_data = pd.concat(df_list, ignore_index=True)

    # Initialize the scaler
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # Apply scaling to all features
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler


def integrate_and_compare(trip_data):
    """
    Perform time-based integration on 'Power_hybrid' and 'Power_data' for trip data.
    """
    # Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Calculate 'Power_hybrid' as the sum of 'Power_phys' and 'y_pred'
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1]

    # Integrate 'Power_data'
    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return hybrid_integral, data_integral


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

    # Step (1): Split the dataset into train and test by file
    all_files = vehicle_files
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    if len(train_files) == 0:
        raise ValueError("No training files were found after splitting. Check file list or split ratio.")
    if len(test_files) == 0:
        raise ValueError("No test files were found after splitting. Check file list or split ratio.")

    # Step (2): 5-Fold cross-validation on train_files
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

        # Process data for fold train/val
        train_data, scaler = process_files(fold_train_files)
        val_data, _ = process_files(fold_val_files, scaler=scaler)

        # Train the model
        X_train = train_data[feature_cols]
        y_train = train_data['Residual']

        X_val = val_data[feature_cols]
        y_val = val_data['Residual']

        model = train_model_linear_regression(X_train, y_train)

        # Make predictions on the validation set
        val_data['y_pred'] = model.predict(X_val)

        # Compute RMSE and MAPE using integration
        val_trip_groups = val_data.groupby('trip_id')

        hybrid_integrals_val, data_integrals_val = [], []
        for _, group in val_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_val.append(hybrid_integral)
            data_integrals_val.append(data_integral)

        mape_val = calculate_mape(np.array(data_integrals_val),
                                  np.array(hybrid_integrals_val))
        rmse_val = calculate_rmse(
            (y_val + val_data['Power_phys']),
            (val_data['y_pred'] + val_data['Power_phys'])
        )

        fold_results.append({
            'fold': fold_num,
            'rmse': rmse_val,
            'mape': mape_val
        })
        fold_models.append(model)
        fold_scalers.append(scaler)

        print(f"[Fold {fold_num}] Validation RMSE = {rmse_val:.4f}, MAPE = {mape_val:.2f}%")

    # Step (3): Select the fold whose Validation RMSE is closest to the median => best model
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

    # Step (4): Evaluate the selected best model on the test set
    test_data, _ = process_files(test_files, scaler=best_scaler)

    X_test = test_data[feature_cols]
    y_test = test_data['Residual']

    test_data['y_pred'] = best_model.predict(X_test)

    test_trip_groups = test_data.groupby('trip_id')
    hybrid_integrals_test, data_integrals_test = [], []
    for _, group in test_trip_groups:
        hybrid_integral, data_integral = integrate_and_compare(group)
        hybrid_integrals_test.append(hybrid_integral)
        data_integrals_test.append(data_integral)

    mape_test = calculate_mape(np.array(data_integrals_test),
                               np.array(hybrid_integrals_test))
    rmse_test = calculate_rmse(
        (y_test + test_data['Power_phys']),
        (test_data['y_pred'] + test_data['Power_phys'])
    )

    print(f"\n[Test Set Evaluation using Best Model (Fold {best_fold})]")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.2f}%")
    print("------------------------------------")

    results = []
    results.append({
        'fold_results': fold_results,
        'best_fold': best_fold,
        'best_model': best_model,
        'test_rmse': rmse_test,
        'test_mape': mape_test
    })

    return results, best_scaler
