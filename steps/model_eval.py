import logging
import mlflow
import pandas as pd

from zenml import step
from zenml.client import Client

from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

experiement_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiement_tracker.name)
def evaluate_model(model: RegressorMixin,
                   x_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   ) -> Tuple[
                        Annotated[float, 'r2'],
                        Annotated[float, 'rmse'],
                        Annotated[float, 'mse']
                   ]:
    '''
    Evaluate the model on the given data
    
    Args:
        df: pd.DataFrame: Data to evaluate the model on
    Returns:
        None
    '''
    try:
        prediction = model.predict(x_test)
        MSE_class = MSE()
        R2_class = R2()
        RMSE_class = RMSE()
        
        mse = MSE_class.calculate_scores(y_test, prediction)
        r2 = R2_class.calculate_scores(y_test, prediction)
        rmse = RMSE_class.calculate_scores(y_test, prediction)
        
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mse', mse)
        
        return r2, rmse, mse
    
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return e        