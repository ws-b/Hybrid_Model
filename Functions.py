import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_mape(y_test, y_pred):
    non_zero_indices = y_test != 0
    y_test_non_zero = y_test[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]

    mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100
    return mape
def calculate_rrmse(y_test, y_pred):
    relative_errors = (y_test - y_pred) / np.mean(np.abs(y_test))
    rrmse = np.sqrt(np.mean(relative_errors ** 2)) * 100
    return rrmse

def calculate_rmse(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_test-y_pred) ** 2))
    return rmse

def integrate_and_compare(trip_data):
    trip_data['time'] = pd.to_datetime(trip_data['time'], format='%Y-%m-%d %H:%M:%S')
    trip_data = trip_data.sort_values(by='time')

    time_seconds = (trip_data['time'] - trip_data['time'].min()).dt.total_seconds().values

    physics_integral = np.trapz(trip_data['Power_phys'].values, time_seconds)
    data_integral = np.trapz(trip_data['Power_data'].values, time_seconds)

    return physics_integral, data_integral
def compute_mape(vehicle_files):
    if not vehicle_files:
        print("No files provided")
        return
    physics_integrals, data_integrals = [], []
    for file in vehicle_files:
        data = pd.read_csv(file)
        physics_integral, data_integral = integrate_and_compare(data)
        physics_integrals.append(physics_integral)
        data_integrals.append(data_integral)

    mape= calculate_mape(np.array(data_integrals), np.array(physics_integrals))
    print(f"MAPE : {mape:.2f}%")

    return mape


def compute_rmse(vehicle_files):
    if not vehicle_files:
        print("No files provided")
        return

    power_phys_all, power_data_all = [], []

    for file in vehicle_files:
        data = pd.read_csv(file)

        if 'Power_phys' not in data.columns or 'Power_data' not in data.columns:
            print(f"Columns 'Power_phys' or 'Power_data' not found in {file}")
            continue

        power_phys = data['Power_phys'].values
        power_data = data['Power_data'].values

        # Store data to aggregate across all files
        power_phys_all.extend(power_phys)
        power_data_all.extend(power_data)

    # Calculate RMSE over the aggregated data
    rmse = np.sqrt(mean_squared_error(power_data_all, power_phys_all))
    print(f"RMSE : {rmse:.2f}")

    return rmse