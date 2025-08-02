import xgboost as xgb
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel
from config import N_TRIALS_OPTUNA

class XGBoostModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('XGBoost', vehicle_name)

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
            param = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e5, log=True), # L1
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1e5, log=True), # L2
                # --- 튜닝하지 않은 파라미터 (기본값 사용) ---
                # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                # 'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                # 'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmse_scores = []
            for train_index, val_index in kf.split(X_full):
                X_train, X_val = X_full.iloc[train_index], X_full.iloc[val_index]
                y_train, y_val = y_full.iloc[train_index], y_full.iloc[val_index]

                model = xgb.XGBRegressor(**param)
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

        return study.best_params

    def train(self, train_data, params, model_type='hybrid'):
        X_train, y_train = self._get_features_and_target(train_data, model_type)
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, test_data, model_type='hybrid'):
        X_test, _ = self._get_features_and_target(test_data, model_type)
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
            
        predictions = self.model.predict(X_test)
        
        if model_type == 'hybrid':
            final_predictions = predictions + test_data['physics_prediction'].values
            return final_predictions
        else:
            return predictions