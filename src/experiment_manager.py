import os
import joblib
import json
import numpy as np
import numpy as np
from config import SELECTED_VEHICLE, RESULTS_DIR

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ExperimentManager:
    def __init__(self, results_dir=RESULTS_DIR):
        self.optuna_dir = os.path.join(results_dir, 'optuna')
        self.log_dir = os.path.join(results_dir, 'logs', SELECTED_VEHICLE)
        self.params_dir = os.path.join(results_dir, 'hyperparameters', SELECTED_VEHICLE)
        self.final_results_path = os.path.join(results_dir, f"{SELECTED_VEHICLE}_final_results.json")
        self.models_dir = os.path.join(results_dir, 'trained_models', SELECTED_VEHICLE)
        
        os.makedirs(self.optuna_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def get_optuna_storage_path(self):
        """Returns the file path for the Optuna study database."""
        return f"sqlite:///{os.path.join(self.optuna_dir, 'optuna_study.db')}"

    def _get_log_filename(self, model_name, size, iteration):
        """
        Generates a standardized log file name FOR EACH MODEL.
        """
        return f"{model_name}_size_{size}_iter_{iteration}.json"

    def is_complete(self, model_name, size, iteration):
        """
        Checks if a SPECIFIC MODEL'S trial has already been completed.
        """
        log_file = self._get_log_filename(model_name, size, iteration)
        return os.path.exists(os.path.join(self.log_dir, log_file))

    def log_result(self, model_name, size, iteration, result_data):
        """
        Logs the results of a single model's trial to a JSON file.
        """
        log_file = self._get_log_filename(model_name, size, iteration)
        log_path = os.path.join(self.log_dir, log_file)

        try:
            with open(log_path, 'w') as f:
                json.dump(result_data, f, indent=4, cls=NpEncoder)
            print(f"      Result for {model_name} logged to {log_path}")
        except IOError as e:
            print(f"      Error logging result to {log_path}: {e}")

    def log_hyperparameters(self, model_name, params):
        """
        Saves the best hyperparameters found for a model.
        """
        params_file = f"{model_name}_best_params.json"
        params_path = os.path.join(self.params_dir, params_file)

        try:
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Best hyperparameters for {model_name} saved to {params_path}")
        except IOError as e:
            print(f"Error saving hyperparameters to {params_path}: {e}")

    def load_hyperparameters(self, model_name):
        """
        Loads the best hyperparameters for a model.
        """
        params_file = f"{model_name}_best_params.json"
        params_path = os.path.join(self.params_dir, params_file)

        if not os.path.exists(params_path):
            return None

        try:
            with open(params_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading hyperparameters from {params_path}: {e}")
            return None
            
    def log_final_results(self, final_results_data):
        """Saves the final validation results on the test set."""
        try:
            with open(self.final_results_path, 'w') as f:
                json.dump(final_results_data, f, indent=4, cls=NpEncoder)
            print(f"Final validation results saved to {self.final_results_path}")
        except IOError as e:
            print(f"Error saving final results: {e}")

    def save_model_and_scaler(self, model_name, model_type, artifacts):
        """학습된 모델과 스케일러를 함께 저장합니다."""
        filename = f"{model_type}_{model_name}_artifacts.joblib"
        path = os.path.join(self.models_dir, filename)
        try:
            joblib.dump(artifacts, path)
            print(f"Artifacts for {model_name} ({model_type}) saved to {path}")
        except Exception as e:
            print(f"Error saving artifacts: {e}")