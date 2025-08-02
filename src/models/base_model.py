from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for all models.
    It defines the common interface for training, prediction, and hyperparameter tuning.
    """
    def __init__(self, model_name, vehicle_name):
        self.model_name = model_name
        self.vehicle_name = vehicle_name
        self.model = None

    @abstractmethod
    def find_best_params(self, train_data, n_trials=100):
        """
        Uses Optuna to find the best hyperparameters for the model.

        Args:
            train_data: The full training dataset.
            n_trials (int): The number of optimization trials.

        Returns:
            A dictionary containing the best hyperparameters.
        """
        pass

    @abstractmethod
    def train(self, train_data, params, model_type='hybrid'):
        """
        Trains the model with the given data and hyperparameters.

        Args:
            train_data: The training data for the current sample.
            params (dict): The hyperparameters for training.
            model_type (str): 'hybrid' or 'ml_only'. Determines the feature set to use.

        Returns:
            The trained model object.
        """
        pass

    @abstractmethod
    def predict(self, test_data, model_type='hybrid'):
        """
        Makes predictions on the test data using the trained model.

        Args:
            test_data: The data to make predictions on.
            model_type (str): 'hybrid' or 'ml_only'.

        Returns:
            A numpy array of predictions.
        """
        pass

    def _get_features_and_target(self, data, model_type):
        """
        Separates features and target variable from the dataframe.
        Handles feature selection based on the model type.
        """
        pass
