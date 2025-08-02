from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna
import json
from .base_model import BaseModel
from config import N_TRIALS_OPTUNA

class MLPModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('MLP', vehicle_name)
        self.scaler = StandardScaler() 

    def _get_features_and_target(self, data, model_type):
        base_features = ['speed', 'acceleration', 'ext_temp']
        rolling_features = [col for col in data.columns if 'roll' in col]
        features_to_use = base_features + rolling_features
        X = data[features_to_use]
        
        if model_type == 'hybrid':
            y = data['target'] - data['physics_prediction']
        else: # ml_only
            y = data['target']
            
        return X, y

    def find_best_params(self, train_data, storage_path):
        X_full, y_full = self._get_features_and_target(train_data, model_type='hybrid')

        def objective(trial):
            hidden_layer_str = trial.suggest_categorical('hidden_layer_sizes', [
                '[50]', '[100]', '[50, 50]', '[100, 50]', '[100, 100]'
            ])
            hidden_layer_sizes = tuple(json.loads(hidden_layer_str))

            param = {
                'hidden_layer_sizes': hidden_layer_sizes,
                'activation': 'relu',
                'solver': 'adam',
                'alpha': trial.suggest_float('alpha', 1e-5, 1e5, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 300, 1000),
                'random_state': 42,
            }
            
            X_scaled = self.scaler.fit_transform(X_full)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmse_scores = []
            for train_index, val_index in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_index], X_scaled[val_index]
                y_train, y_val = y_full.iloc[train_index], y_full.iloc[val_index]

                model = MLPRegressor(**param)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

            return np.mean(rmse_scores)

        study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.model_name}-{self.vehicle_name}-tuning",
            storage=storage_path,
            load_if_exists=True
        )
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA)

        best_params = study.best_params
        best_params['hidden_layer_sizes'] = tuple(json.loads(best_params['hidden_layer_sizes']))
        best_params['activation'] = 'relu'

        return best_params

    def train(self, train_data, params, model_type='hybrid'):
        X_train, y_train = self._get_features_and_target(train_data, model_type)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = MLPRegressor(**params)
        self.model.fit(X_train_scaled, y_train)
        
        return self.model, self.scaler

    def predict(self, test_data, model_type='hybrid'):
        X_test, _ = self._get_features_and_target(test_data, model_type)
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test_scaled)
        
        if model_type == 'hybrid':
            final_predictions = predictions + test_data['physics_prediction'].values
            return final_predictions
        else: # ml_only
            return predictions