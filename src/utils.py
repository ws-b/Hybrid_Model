import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_physics_rmse(data):
    """
    Computes the Root Mean Squared Error (RMSE) for the physics-based model.
    
    This function assumes the physics model's predictions are in a column
    named 'physics_prediction' and the actual values are in 'target'.
    
    Args:
        data (pd.DataFrame): A DataFrame containing the predictions and actuals.
        
    Returns:
        float: The calculated RMSE value.
    """
    if 'physics_prediction' not in data.columns or 'target' not in data.columns:
        # Returning a high value or raising an error might be appropriate
        # depending on how missing predictions should be handled.
        print("Warning: Physics model predictions or target not found.")
        return np.inf
        
    # Ensure there are no NaN values which would result in an error
    cleaned_data = data.dropna(subset=['physics_prediction', 'target'])
    
    if cleaned_data.empty:
        print("Warning: No valid data points for physics RMSE calculation.")
        return np.inf

    rmse = np.sqrt(mean_squared_error(cleaned_data['target'], cleaned_data['physics_prediction']))
    return rmse

def scale_hyperparameters(params, adjustment_factor):
    """
    Scales the regularization hyperparameters based on the sample size.
    
    Args:
        params (dict): The dictionary of hyperparameters.
        adjustment_factor (float): The factor to scale the parameters by (N_total / size).
        
    Returns:
        dict: The dictionary with scaled hyperparameters.
    """
    scaled_params = params.copy()
    
    # Parameters to scale (add others if needed)
    # These are common names for L1/L2 regularization in various libraries
    params_to_scale = ['reg_alpha', 'reg_lambda', 'alpha', 'l2', 'weight_decay']
    
    for param, value in scaled_params.items():
        if param in params_to_scale:
            # We don't scale learning rate (eta)
            if param not in ['eta', 'learning_rate']:
                scaled_params[param] = value * adjustment_factor
                
    return scaled_params

def compute_energy_mape(test_data, predictions):
    """
    Computes the Mean Absolute Percentage Error (MAPE) for total trip energy
    using numerical integration (trapezoidal rule) for robustness.
    
    Args:
        test_data (pd.DataFrame): DataFrame with 'trip_id', 'time', and 'target'.
        predictions (np.array): Numpy array of power predictions.
        
    Returns:
        float: The calculated MAPE value for total energy.
    """
    if not all(col in test_data.columns for col in ['trip_id', 'time', 'target']):
        raise ValueError("test_data must contain 'trip_id', 'time', and 'target' columns.")

    eval_df = test_data[['trip_id', 'time', 'target']].copy()
    eval_df['prediction'] = predictions
    # 시간 순으로 정렬 보장
    eval_df.sort_values(by=['trip_id', 'time'], inplace=True)

    # 수치 적분을 위한 에너지 계산
    total_energies = []
    for trip_id in eval_df['trip_id'].unique():
        trip_df = eval_df[eval_df['trip_id'] == trip_id]
        
        # 각 포인트 사이의 시간 차이(delta_t)를 초 단위로 계산
        delta_t = trip_df['time'].diff().dt.total_seconds().values
        
        # 실제 전력과 예측 전력 값
        actual_power = trip_df['target'].values
        predicted_power = trip_df['prediction'].values
        
        # 사다리꼴 공식을 이용한 에너지 적분
        # 에너지 = (P1 + P2) / 2 * delta_t
        # np.trapz(y, x) 함수는 이 계산을 효율적으로 수행함
        actual_energy = np.trapz(y=actual_power, x=trip_df['time'].astype(np.int64) // 10**9)
        predicted_energy = np.trapz(y=predicted_power, x=trip_df['time'].astype(np.int64) // 10**9)
        
        total_energies.append({
            'trip_id': trip_id,
            'actual_energy': actual_energy,
            'predicted_energy': predicted_energy
        })

    energy_summary = pd.DataFrame(total_energies)
    
    # 0으로 나누는 것을 방지
    energy_summary = energy_summary[energy_summary['actual_energy'] != 0]

    if energy_summary.empty:
        return np.inf

    # 각 trip의 MAPE 계산
    percentage_errors = np.abs(
        (energy_summary['actual_energy'] - energy_summary['predicted_energy']) / energy_summary['actual_energy']
    )
    
    mape = np.mean(percentage_errors) * 100
    return mape