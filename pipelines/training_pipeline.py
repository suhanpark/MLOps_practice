from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.model_eval import evaluate_model
from steps.config import model_name

@pipeline
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, y_train, x_test, y_test, model_name)
    r2, rmse, mse = evaluate_model(model, x_test, y_test)
    