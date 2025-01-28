import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    '''
    Abstract class for evaluation
    '''
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''
        Calculate the scores of the model
        
        Args:
            model: Model: Model to evaluate
            x_test: pd.DataFrame: Testing features
            y_test: pd.Series: Testing target
        Returns:    
            None
        '''
        pass
    
class MSE(Evaluation):
    '''
    Mean squared error
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''
        Calculate the mean squared error
        
        Args:
            y_true: np.ndarray: True values
            y_pred: np.ndarray: Predicted values
        Returns:
            float: Mean squared error
        '''
        try:
            logging.info("Calculating mean squared error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean squared error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating mean squared error: {e}")
            return e
        
class R2(Evaluation):
    '''
    R squared
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''
        Calculate the R squared
        
        Args:
            y_true: np.ndarray: True values
            y_pred: np.ndarray: Predicted values
        Returns:
            float: R squared
        '''
        try:
            logging.info("Calculating R squared")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R squared: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R squared: {e}")
            return e
        
class RMSE(Evaluation):
    '''
    Root mean squared error
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        '''
        Calculate the root mean squared error
        
        Args:
            y_true: np.ndarray: True values
            y_pred: np.ndarray: Predicted values
        Returns:
            float: Root mean squared error
        '''
        try:
            logging.info("Calculating root mean squared error")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root mean squared error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating root mean squared error: {e}")
            return e