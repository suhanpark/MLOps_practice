import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    '''
    Abstract class for data strategies
    '''
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreprocessStrategy(DataStrategy):
    '''
    Data preprocessing strategy
    '''
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Preprocess the data
        '''
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data['review_comment_message'].fillna('No review', inplace=True)
            
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            return e     

class DataDivideStrategy(DataStrategy):
    '''
    Data division strategy
    '''
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Divide the data into features and target
        '''
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error dividing data: {e}")
            return e       
        
        
class DataCleaning:
    '''
    Data cleaning class
    '''
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        '''
        Args:
            data: pd.DataFrame: Data to clean
            strategy: DataStrategy: Strategy to clean the data
        '''
        self.data = data
        self.strategy = strategy
        
    def handle_data(self):
        '''
        Clean the data
        '''
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            return e
