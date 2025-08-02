from sklearn.linear_model import LinearRegression
import numpy as np
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self, vehicle_name):
        super().__init__('MLR', vehicle_name)

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
        print("LinearRegression does not require hyperparameter tuning.")
        return {}

    def train(self, train_data, params, model_type='hybrid'):
        X_train, y_train = self._get_features_and_target(train_data, model_type)
        self.model = LinearRegression(**params)
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
        else: # ml_only
            return predictions