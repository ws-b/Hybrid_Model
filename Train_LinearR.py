import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Functions import calculate_rmse, calculate_mape
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.integrate import cumulative_trapezoid

# This code contains comments generated using ChatGPT.
# ----------------------------
# Data Processing Functions
# ----------------------------
def process_single_file(file):
    """
    Processes a single CSV file to compute the residual and select relevant columns.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            data['Residual'] = data['Power_data'] - data['Power_phys']
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Residual', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    Processes multiple CSV files in parallel, calculates rolling statistics, and applies feature scaling.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # Convert km/h to m/s
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
        future_to_file = {executor.submit(process_single_file, file): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                data = future.result()
                if data is not None:
                    # Convert the 'time' column to datetime
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # Add trip_id to distinguish trips
                    data['trip_id'] = files.index(file)

                    # Calculate rolling statistics with a window size of 5
                    data['mean_accel_10'] = data['acceleration'].rolling(window=5).mean().bfill()
                    data['std_accel_10'] = data['acceleration'].rolling(window=5).std().bfill()
                    data['mean_speed_10'] = data['speed'].rolling(window=5).mean().bfill()
                    data['std_speed_10'] = data['speed'].rolling(window=5).std().bfill()

                    df_list.append(data)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found. Please check the input files and try again.")

    full_data = pd.concat(df_list, ignore_index=True)

    # 스케일러가 없는 경우 새로 생성
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
    Integrates 'Power_hybrid' and 'Power_data' over time using the trapezoidal rule.
    """
    # Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Integrate 'Power_phys + y_pred' using the trapezoidal rule
    trip_data['Power_hybrid'] = trip_data['Power_phys'] + trip_data['y_pred']
    hybrid_cum_integral = cumulative_trapezoid(trip_data['Power_hybrid'].values, time_seconds, initial=0)
    hybrid_integral = hybrid_cum_integral[-1]

    # Integrate 'Power_data' using the trapezoidal rule
    data_cum_integral = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cum_integral[-1]

    return hybrid_integral, data_integral


def train_model_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ----------------------------
# Cross-validation and Model Training
# ----------------------------
def cross_validate(files):
    """
    Performs 5-fold cross-validation on the given list of CSV files,
    trains a Linear Regression model, calculates metrics, and selects the best model.
    """
    model_name = "LinearRegression"
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []

    # K-fold split on the list of files
    for fold_num, (train_index, test_index) in enumerate(kf.split(files), start=1):
        train_files = [files[i] for i in train_index]
        test_files = [files[i] for i in test_index]

        # Process training and testing data
        train_data, scaler = process_files(train_files)
        test_data, _ = process_files(test_files, scaler=scaler)

        feature_cols = [
            'speed', 'acceleration', 'ext_temp',
            'mean_accel_10', 'std_accel_10',
            'mean_speed_10', 'std_speed_10'
        ]

        # Prepare training and testing data
        X_train = train_data[feature_cols]
        y_train = train_data['Residual']

        X_test = test_data[feature_cols]
        y_test = test_data['Residual']

        # Train the model
        model = train_model_linear_regression(X_train, y_train)

        # Make predictions
        train_data['y_pred'] = model.predict(X_train)
        test_data['y_pred'] = model.predict(X_test)

        # Integrate by trip for training data
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        hybrid_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_train.append(hybrid_integral)
            data_integrals_train.append(data_integral)

        # Calculate MAPE for the training data
        mape_train = calculate_mape(
            np.array(data_integrals_train),
            np.array(hybrid_integrals_train)
        )

        # Integrate by trip for testing data
        hybrid_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            hybrid_integral, data_integral = integrate_and_compare(group)
            hybrid_integrals_test.append(hybrid_integral)
            data_integrals_test.append(data_integral)

        # Calculate MAPE for the testing data
        mape_test = calculate_mape(
            np.array(data_integrals_test),
            np.array(hybrid_integrals_test)
        )

        # Calculate RMSE
        rmse = calculate_rmse(
            (y_test + test_data['Power_phys']),
            (test_data['y_pred'] + test_data['Power_phys'])
        )

        # Store results
        results.append({
            'fold': fold_num,
            'rmse': rmse,
            'test_mape': mape_test,
            'best_params': None  # No hyperparameters for standard Linear Regression
        })
        models.append(model)

        # Print fold results
        print(f"--- Fold {fold_num} Results ---")
        print(f"RMSE: {rmse:.2f}")
        print(f"Train - MAPE: {mape_train:.2f}%")
        print(f"Test - MAPE: {mape_test:.2f}%")
        print("---------------------------\n")

    # Once all folds are complete, select the best model
    if len(results) == kf.get_n_splits():
        rmse_values = [result['rmse'] for result in results]
        median_rmse = np.median(rmse_values)
        closest_index = np.argmin(np.abs(np.array(rmse_values) - median_rmse))
        best_model = models[closest_index]
        selected_fold = results[closest_index]['fold']
        print(f"Selected Fold {selected_fold} as Best Model with RMSE: {rmse_values[closest_index]:.4f}")
    else:
        best_model = None
        print("No models available to select as best_model.")

    return results, scaler
