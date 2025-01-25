import logging

import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    '''
    Evaluate the model on the given data
    
    Args:
        df: pd.DataFrame: Data to evaluate the model on
    Returns:
        None
    '''
    pass