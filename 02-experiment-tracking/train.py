import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Additions
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("duration-prediction-training")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run_train():

    X_train, y_train = load_pickle("output/train.pkl")
    X_val, y_val = load_pickle("output/val.pkl")

    mlflow.autolog()
    
    with mlflow.start_run():
        
        
        depth = 10
        random_state = 0
        
        rf = RandomForestRegressor(max_depth=depth, random_state=random_state)
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
