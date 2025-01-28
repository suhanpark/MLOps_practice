import logging
from numpy import divide
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'x_train'],
    Annotated[pd.DataFrame, 'x_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    '''
    Clean and divide the data
    
    Args:
        df: pd.DataFrame: Data to clean and divide
    Returns:
        x_train: pd.DataFrame: Training data
        x_test: pd.DataFrame: Testing data
        y_train: pd.DataSeries: Training target
        y_test: pd.DataSeries: Testing target
    '''
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaned and divided")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        return e
