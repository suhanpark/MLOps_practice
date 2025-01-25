import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    '''
    Train a model on the given data
    
    Args:
        df: pd.DataFrame: Data to train the model on
    Returns:
        None
    '''
    logging.info("Training the model")
    pass