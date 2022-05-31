from ast import arg
import logging
import numpy as np
import pandas as pd
import pathlib
import catboost
import shap
import json
import argparse

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn import datasets

# from constants import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    LabelEncoder, 
    OneHotEncoder
)
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import get_logger, calc_regression_metrics

logger = get_logger(name= pathlib.Path(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument("--tracking_uri", type=str, default='https://mlflow.healthcare.com/')
    parser.add_argument("--experiment_name", type=str,
                        default="mlflow_test")
    parser.add_argument('--user_arn', type=str, default='rbhende')
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--loss_function', type=str, default='RMSE')
    parser.add_argument('--target', type=str, default='target')
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data/dataset.csv'
    )

    args, _ = parser.parse_known_args()

    logger.info(f"Tracking URL: {args.tracking_uri}")
    logger.info(f"Experiment Name: {args.experiment_name}")

    ## Initialize MLflow params
    try:
        mlflow.create_experiment(args.experiment_name,'s3://hc-prd-mlflow-bucket')
        logger.info(f"Experiment {args.experiment_name} is successfully created.")
    except MlflowException as ex:
        logger.exception(f"Error creating experiment {ex}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    ## Post-process the Dataset
    data = pd.read_csv(args.data_path, low_memory= False)
    # Replace Null values with 0 for numeric columns and N/A for categorical
    categorical_columns = []
    numeric_columns = []
    for col in data.columns:
        if data[col].dtype in ['i', 'f', int, float]:
            numeric_columns.append(col)
            data[col] = data[col].fillna(0)
        elif data[col].dtype in ['O', 'S', 'a']:
            categorical_columns.append(col)
            data[col] = data[col].fillna('N/A')
        else:
            data[col] = data[col].fillna('N/A')
    numeric_columns.remove(args.target) # Target Variable

    ## Train-Test Split
    y = data[args.target]
    X = data.drop(columns=[args.target])
    X_train, X_test, y_train, y_test = train_test_split(
        X= X,
        y= y,
        test_size = 0.2, 
        random_state=0
    )

    logger.info(
        f"Splitted input dataset shapes are: \
            X_train= {X_train.shape}, y_train= {y_train.shape},\
                X_test= {X_test.shape}, y_test= {y_test.shape}"
    )

    ## Define model parameters
    model_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "loss_function": args.loss_function
    }

    ## Initialize the CatBoost model
    model = catboost.CatBoostRegressor(**model_params)
    ## Train the model
    try:
        model.fit(
            X= X_train, 
            y= y_train,
            cat_features= categorical_columns,
            plot= True
        )
        logger.info("CatBoost model is trained.")
    except Exception as e:
        logger.exception(f"Exception {e} occured during training CatBoost model.")
    
    ## Calculate Regression Metrics
    try:
        regr_metrics = calc_regression_metrics(
            model= model,
            X_train= X_train,
            y_train= y_train,
            X_test= X_test,
            y_test= y_test
        )
        logger.info(f"Regression model preformance metrics are: {json.dumps(regr_metrics)}")
    except Exception as e:
        logger.exception(f"Exception {e} occured during calculating regression metrics.")

    signature = infer_signature(X_test, model.predict_proba(X_test))

    with mlflow.start_run() as run:
        mlflow.set_tag('user_arn', args.user_arn)
        mlflow.sklearn.log_model(model, 'model', signature=signature)
        mlflow.log_params(model_params)
        mlflow.log_metrics(regr_metrics)

        experiment = mlflow.get_experiment_by_name(args.experiment_name)

        logger.info(f'experiment_id={experiment.experiment_id}')
        logger.info(f'artifact_location={experiment.artifact_location}')
        logger.info(f'tags={experiment.tags}')
        logger.info(f'lifecycle_stage={experiment.lifecycle_stage}')
        logger.info(f'artifact_uri={mlflow.get_artifact_uri()}')
        logger.info(f'run_id={mlflow.active_run().info.run_id}')
    
    mlflow.end_run()
    logger.info('done')


    







