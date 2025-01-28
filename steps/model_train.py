import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from src.model_dev import LinearRegressionModel
from .config import model_name

experiment_tracker = Client().active_stack.experiment_tracker

@step
def train_model(
                x_train: pd.DataFrame,
                y_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_test: pd.DataFrame, 
                model_name: str) -> RegressorMixin:
    '''
    Train a model on the given data
    
    Args:
        df: pd.DataFrame: Data to train the model on
    Returns:
        None
    '''
    try:
        model = None
        if model_name == 'LinearRegressionModel':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {model_name} not found")
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return e
    