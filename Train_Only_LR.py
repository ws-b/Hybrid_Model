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
    Reads a single CSV file and extracts relevant columns for further analysis.

    Parameters:
        file (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: A DataFrame containing the time, speed, acceleration,
            external temperature, Power_phys, and Power_data columns. Returns
            None if the file cannot be processed.
    """
    try:
        data = pd.read_csv(file)
        if 'Power_phys' in data.columns and 'Power_data' in data.columns:
            return data[['time', 'speed', 'acceleration', 'ext_temp', 'Power_phys', 'Power_data']]
    except Exception as e:
        print(f"Error processing file {file}: {e}")
    return None


def process_files(files, scaler=None):
    """
    Processes multiple CSV files in parallel, calculates rolling statistics,
    and applies feature scaling.

    Parameters:
        files (list): List of CSV file paths.
        scaler (MinMaxScaler, optional): An optional scaler to use for
            consistency between train/test data. If None, a new scaler is
            created from standard ranges.

    Returns:
        (pd.DataFrame, MinMaxScaler): A tuple of:
            - A DataFrame with all processed data (including rolling
              statistics and scaled features).
            - The MinMaxScaler object used for feature scaling.

    Raises:
        ValueError: If no valid data files are found.
    """
    SPEED_MIN = 0 / 3.6
    SPEED_MAX = 230 / 3.6  # Convert km/h to m/s
    ACCELERATION_MIN = -15  # m/s^2
    ACCELERATION_MAX = 9    # m/s^2
    TEMP_MIN = -30
    TEMP_MAX = 50

    # Columns to be scaled
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
                    # Convert the 'time' column to datetime format
                    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')

                    # Add trip_id to differentiate between trips
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

    # Create a new scaler if one is not provided
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(pd.DataFrame([
            [SPEED_MIN, ACCELERATION_MIN, TEMP_MIN, 0, 0, 0, 0],
            [SPEED_MAX, ACCELERATION_MAX, TEMP_MAX, 1, 1, 1, 1]
        ], columns=feature_cols))

    # Apply scaling to all feature columns
    full_data[feature_cols] = scaler.transform(full_data[feature_cols])

    return full_data, scaler


def integrate_and_compare(trip_data):
    """
    Integrates 'y_pred' and 'Power_data' over time using the trapezoidal rule
    and compares their total energy values.

    Parameters:
        trip_data (pd.DataFrame): DataFrame containing the data for a specific trip_id.

    Returns:
        tuple: (ml_integral, data_integral)
            - ml_integral (float): Total energy (integral) of y_pred.
            - data_integral (float): Total energy (integral) of Power_data.
    """
    # Sort by 'time'
    trip_data = trip_data.sort_values(by='time')

    # Convert 'time' to seconds
    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    # Integrate 'y_pred' using the cumulative trapezoidal rule
    ml_cumulative = cumulative_trapezoid(trip_data['y_pred'].values, time_seconds, initial=0)
    ml_integral = ml_cumulative[-1]

    # Integrate 'Power_data' using the cumulative trapezoidal rule
    data_cumulative = cumulative_trapezoid(trip_data['Power_data'].values, time_seconds, initial=0)
    data_integral = data_cumulative[-1]

    return ml_integral, data_integral


def train_model_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values for training.

    Returns:
        LinearRegression: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ----------------------------
# Cross-Validation & Model Training
# ----------------------------
def cross_validate(files):
    """
    Performs K-fold cross-validation on a given list of CSV files using Linear Regression.

    Parameters:
        files (list): List of CSV file paths.

    Returns:
        (list, MinMaxScaler): A tuple containing:
            - A list of dictionaries with fold-level results (RMSE, MAPE).
            - The MinMaxScaler used for feature scaling.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    models = []

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

        # Prepare training and testing sets
        X_train = train_data[feature_cols]
        y_train = train_data['Power_data']

        X_test = test_data[feature_cols]
        y_test = test_data['Power_data']

        # Train the model
        model = train_model_linear_regression(X_train, y_train)

        # Predict on both training and test sets
        train_data['y_pred'] = model.predict(X_train)
        test_data['y_pred'] = model.predict(X_test)

        # Calculate integrals for each trip in the training set
        train_trip_groups = train_data.groupby('trip_id')
        test_trip_groups = test_data.groupby('trip_id')

        ml_integrals_train, data_integrals_train = [], []
        for _, group in train_trip_groups:
            ml_integral, data_integral = integrate_and_compare(group)
            ml_integrals_train.append(ml_integral)
            data_integrals_train.append(data_integral)

        # Compute MAPE for the training set
        mape_train = calculate_mape(
            np.array(data_integrals_train),
            np.array(ml_integrals_train)
        )

        # Calculate integrals for each trip in the test set
        ml_integrals_test, data_integrals_test = [], []
        for _, group in test_trip_groups:
            ml_integral, data_integral = integrate_and_compare(group)
            ml_integrals_test.append(ml_integral)
            data_integrals_test.append(data_integral)

        # Compute MAPE for the test set
        mape_test = calculate_mape(
            np.array(data_integrals_test),
            np.array(ml_integrals_test)
        )

        # Compute RMSE for the test set
        rmse = calculate_rmse(y_test, test_data['y_pred'])

        # Store results
        results.append({
            'fold': fold_num,
            'rmse': rmse,
            'test_mape': mape_test,
        })
        models.append(model)

        # Print fold results
        print(f"--- Fold {fold_num} Results ---")
        print(f"RMSE : {rmse:.2f}")
        print(f"Train - MAPE: {mape_train:.2f}%")
        print(f"Test  - MAPE: {mape_test:.2f}%")
        print("---------------------------\n")

    # After completing all folds, select the best model based on median RMSE
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
