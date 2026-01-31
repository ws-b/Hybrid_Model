import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from config import (
    SELECTED_VEHICLE, SAMPLING_SIZES, SAMPLING_ITERATIONS,
    MODELS_TO_RUN, MODELS_TO_TUNE, MODELS_TO_SKIP
)
from src.data_loader import DataLoader
from src.experiment_manager import ExperimentManager
from src.utils import compute_physics_rmse, scale_hyperparameters, compute_energy_mape

from src.models.xgboost_model import XGBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.linear_regression_model import LinearRegressionModel

def run_experiment():
    print(f"Starting experiment for vehicle: {SELECTED_VEHICLE}")
    if MODELS_TO_SKIP:
        print(f"Skipping the following models as requested: {MODELS_TO_SKIP}")

    data_loader = DataLoader()
    full_dataset = data_loader.get_full_dataset()

    print("Splitting data into training (80%) and test (20%) sets...")
    all_trip_ids = full_dataset['trip_id'].unique()
    train_trip_ids, test_trip_ids = train_test_split(all_trip_ids, test_size=0.2, random_state=42)

    full_train_set = full_dataset[full_dataset['trip_id'].isin(train_trip_ids)].copy()
    test_set = full_dataset[full_dataset['trip_id'].isin(test_trip_ids)].copy()

    N_total_train = len(train_trip_ids)
    print(f"Total trips: {len(all_trip_ids)}, Training trips: {N_total_train}, Test trips: {len(test_trip_ids)}")

    # --- 튜닝용 데이터 크기 제한 로직 ---
    MAX_TUNING_TRIPS = 10000
    tuning_set = full_train_set.copy()
    if N_total_train > MAX_TUNING_TRIPS:
        print(f"\nTraining set has {N_total_train} trips. Sampling down to {MAX_TUNING_TRIPS} for hyperparameter tuning.")
        tuning_trip_ids = pd.Series(train_trip_ids).sample(n=MAX_TUNING_TRIPS, random_state=42).values
        tuning_set = full_train_set[full_train_set['trip_id'].isin(tuning_trip_ids)].copy()

    N_tuning_set = len(tuning_set['trip_id'].unique())
    print(f"Using {N_tuning_set} trips for hyperparameter tuning.")

    experiment_manager = ExperimentManager()
    optuna_storage_path = experiment_manager.get_optuna_storage_path()

    models = {
        "XGBoost": XGBoostModel(SELECTED_VEHICLE),
        "RandomForest": RandomForestModel(SELECTED_VEHICLE),
        "MLR": LinearRegressionModel(SELECTED_VEHICLE),
    }

    # --- 1단계: 하이퍼파라미터 튜닝 ---
    print("\n--- Step 1: Hyperparameter Tuning ---")
    for model_name in tqdm(MODELS_TO_TUNE, desc="Tuning Models"):
        if model_name in MODELS_TO_SKIP:
            print(f"Skipping tuning for {model_name}.")
            continue

        print(f"Tuning {model_name}...")
        best_params = experiment_manager.load_hyperparameters(model_name)
        if not best_params:
            model_instance = models[model_name]
            best_params = model_instance.find_best_params(tuning_set, storage_path=optuna_storage_path)
            experiment_manager.log_hyperparameters(model_name, best_params)
        print(f"Best params for {model_name}: {best_params}")

    # --- 2단계: 샘플링 기반 평가 루프 ---
    sampling_sizes_final = [s for s in SAMPLING_SIZES if s < N_total_train]
    if N_total_train not in sampling_sizes_final:
        sampling_sizes_final.append(N_total_train)

    print("\n--- Step 2: Running Experiment Loop with Cross-Validation ---")
    for size in tqdm(sampling_sizes_final, desc="Sampling Sizes"):
        if size < 5:
            print(f"Sample size {size} is too small for 5-fold CV. Skipping.")
            continue

        num_iterations = SAMPLING_ITERATIONS.get(size, 1)
        print(f"\nProcessing sample size: {size} (Iterations: {num_iterations})")

        for i in tqdm(range(num_iterations), desc=f"Size {size} Iterations", leave=False):
            iteration_num = i + 1
            print(f"  Iteration {iteration_num}/{num_iterations}")

            sampled_trip_ids = train_trip_ids if size == N_total_train else pd.Series(train_trip_ids).sample(n=size, random_state=iteration_num).values

            for model_name in MODELS_TO_RUN:
                if model_name in MODELS_TO_SKIP:
                    continue

                if experiment_manager.is_complete(model_name, size, iteration_num):
                    print(f"    {model_name} (size {size}, iter {iteration_num}) already completed. Skipping.")
                    continue

                print(f"    Running {model_name} with 5-Fold Cross-Validation...")
                model_instance = models[model_name]
                best_params = experiment_manager.load_hyperparameters(model_name)

                adjusted_params = best_params.copy()
                if model_name in ["XGBoost"]:
                    adjustment_factor = N_tuning_set / size
                    adjusted_params = scale_hyperparameters(best_params, adjustment_factor)

                kf = KFold(n_splits=5, shuffle=True, random_state=iteration_num)
                hybrid_rmses, ml_only_rmses, physics_rmses = [], [], []

                for fold_train_ids_idx, fold_val_ids_idx in kf.split(sampled_trip_ids):
                    fold_train_trip_ids = sampled_trip_ids[fold_train_ids_idx]
                    fold_val_trip_ids = sampled_trip_ids[fold_val_ids_idx]

                    fold_train_data = full_train_set[full_train_set['trip_id'].isin(fold_train_trip_ids)]
                    fold_val_data = full_train_set[full_train_set['trip_id'].isin(fold_val_trip_ids)]

                    model_instance.train(fold_train_data, adjusted_params, model_type='hybrid')
                    predictions = model_instance.predict(fold_val_data, model_type='hybrid')
                    hybrid_rmses.append(np.sqrt(mean_squared_error(fold_val_data['target'], predictions)))

                    model_instance.train(fold_train_data, adjusted_params, model_type='ml_only')
                    predictions = model_instance.predict(fold_val_data, model_type='ml_only')
                    ml_only_rmses.append(np.sqrt(mean_squared_error(fold_val_data['target'], predictions)))

                    physics_rmses.append(compute_physics_rmse(fold_val_data))

                log_data = {
                    "sample_size": size, "iteration": iteration_num,
                    "sampled_trip_ids": sampled_trip_ids.tolist(),
                    "results": {
                        "hybrid_rmse": np.mean(hybrid_rmses), "ml_only_rmse": np.mean(ml_only_rmses),
                        "physics_rmse_on_sample": np.mean(physics_rmses),
                        "hybrid_rmse_std": np.std(hybrid_rmses), "ml_only_rmse_std": np.std(ml_only_rmses),
                    }
                }
                experiment_manager.log_result(model_name, size, iteration_num, log_data)

    print("\nExperiment loop completed.")

    # --- 3단계: 최종 모델 검증 (Test Set) ---
    print("\n" + "="*80)
    print("--- Step 3: Final Validation on Unseen Test Set ---")
    print("="*80)

    final_results = {}

    physics_power_rmse = compute_physics_rmse(test_set)
    physics_energy_mape = compute_energy_mape(test_set, test_set['physics_prediction'])
    final_results["Physics_Based"] = {"Power_RMSE": physics_power_rmse, "Energy_MAPE": physics_energy_mape}
    print(f"Physics-Based Model:\n  - Power RMSE: {physics_power_rmse:.2f}\n  - Energy MAPE: {physics_energy_mape:.2f} %")

    for model_name in tqdm(MODELS_TO_RUN, desc="Final Validation"):
        if model_name in MODELS_TO_SKIP:
            print(f"\nSkipping final validation for {model_name}.")
            continue

        print(f"\nValidating {model_name} on the test set...")
        model_instance = models[model_name]
        best_params = experiment_manager.load_hyperparameters(model_name)
        if best_params is None: continue

        # Hybrid 모델 학습, 저장 및 평가
        try:
            trained_artifacts = model_instance.train(full_train_set, best_params, model_type='hybrid')

            artifacts_to_save = {}
            model = trained_artifacts
            artifacts_to_save = {'model': model}
            experiment_manager.save_model_and_scaler(model_name, 'hybrid', artifacts_to_save)

            predictions = model_instance.predict(test_set, model_type='hybrid')
            power_rmse = np.sqrt(mean_squared_error(test_set['target'], predictions))
            energy_mape = compute_energy_mape(test_set, predictions)
            final_results[f"Hybrid_{model_name}"] = {"Power_RMSE": power_rmse, "Energy_MAPE": energy_mape}
            print(f"  Hybrid {model_name}:\n    - Power RMSE: {power_rmse:.2f}\n    - Energy MAPE: {energy_mape:.2f} %")
        except Exception as e:
            print(f"    Error during final validation of Hybrid {model_name}: {e}")

        # OnlyML 모델 학습, 저장 및 평가
        try:
            trained_artifacts_ml = model_instance.train(full_train_set, best_params, model_type='ml_only')

            artifacts_to_save_ml = {}
            model_ml = trained_artifacts_ml
            artifacts_to_save_ml = {'model': model_ml}
            experiment_manager.save_model_and_scaler(model_name, 'ml_only', artifacts_to_save_ml)

            predictions = model_instance.predict(test_set, model_type='ml_only')
            power_rmse = np.sqrt(mean_squared_error(test_set['target'], predictions))
            energy_mape = compute_energy_mape(test_set, predictions)
            final_results[f"OnlyML_{model_name}"] = {"Power_RMSE": power_rmse, "Energy_MAPE": energy_mape}
            print(f"  OnlyML {model_name}:\n    - Power RMSE: {power_rmse:.2f}\n    - Energy MAPE: {energy_mape:.2f} %")
        except Exception as e:
            print(f"    Error during final validation of OnlyML {model_name}: {e}")

    experiment_manager.log_final_results(final_results)

    print("\nExperiment completed. To see the final results, run 'python process_results.py' and 'python plot_final_figure.py'")


if __name__ == "__main__":
    run_experiment()