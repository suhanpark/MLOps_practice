import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    '''
    Abstract class for models
    '''
    @abstractmethod
    def train(self, x_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, x_test, y_test):
        pass
    
class LinearRegressionModel(Model):
    '''
    Linear regression model
    '''
        
    def train(self,x_train, y_train, **kwargs):
        '''
        Train the linear regression model
        '''
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Training the linear regression model")
            return reg
        except Exception as e:
            logging.error(f"Error training the linear regression model: {e}")
            return e
    
    def predict(self):
        '''
        Predict using the linear regression model
        '''
        logging.info("Predicting using the linear regression model")
        pass