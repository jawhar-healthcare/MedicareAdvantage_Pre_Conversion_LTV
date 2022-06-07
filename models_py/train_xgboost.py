from ast import arg
import logging
import numpy as np
import pandas as pd
import pathlib
from xgboost import XGBRegressor

import shap
import json
import argparse

import mlflow

# import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import (
#     MinMaxScaler,
#     StandardScaler,
#     LabelEncoder,
#     OneHotEncoder,
# )

# import seaborn as sns
import matplotlib.pyplot as plt

from utils.load_config_file import load_config_file
from utils.utils import get_logger, calc_regression_metrics
from warnings import filterwarnings

filterwarnings("ignore")

logger = get_logger(name=pathlib.Path(__file__))

CONFIG_PATH = "config/config.ini"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MLflow related parameters
    parser.add_argument(
        "--tracking_uri", type=str, default="https://mlflow.healthcare.com/"
    )
    parser.add_argument("--experiment_name", type=str, default="rb_test1")
    parser.add_argument("--user_arn", type=str, default="rbhende")

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--gamma", type=int, default=0)
    parser.add_argument("--max_depth", type=str, default=10)
    parser.add_argument("--target", type=str, default="LTV")
    parser.add_argument(
        "--data_path", type=str, default="data/with_zcta/ma_ltv_merged.csv"
    )

    args, _ = parser.parse_known_args()

    logger.info(f"Tracking URL: {args.tracking_uri}")
    logger.info(f"Experiment Name: {args.experiment_name}")

    ## Initialize MLflow params
    try:
        mlflow.create_experiment(args.experiment_name, "s3://hc-prd-mlflow-bucket")
        logger.info(f"Experiment {args.experiment_name} is successfully created.")
    except MlflowException as ex:
        logger.exception(f"Error creating experiment {ex}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    ## Post-process the Dataset
    data = pd.read_csv(args.data_path, low_memory=False)
    config = load_config_file(config_path=CONFIG_PATH)
    ## Remove unwanted features
    unwanted_features = config["unwanted_features"]

    ## Remove any post-conversion data features; replace nulls with 0 or N/A
    unwanted_features = (
        unwanted_features
        + [p for p in data.columns if "post_raw" in p.lower()]
        + [c for c in data.columns if len(data[c].fillna(0).unique()) == 1]
    )

    data = data.drop(columns=unwanted_features)

    # replace Null values with 0 for numeric columns and N/A for categorical
    categorical_columns = []
    numeric_columns = []
    for col in data.columns:
        if data[col].dtype in ["i", "f", int, float]:
            numeric_columns.append(col)
            data[col].fillna(0, inplace=True)
        elif data[col].dtype in ["O", "S", "a"]:
            categorical_columns.append(col)
            data[col].fillna("N/A", inplace=True)
        else:
            data[col].fillna("N/A", inplace=True)
    numeric_columns.remove(args.target)  # Target Variable

    data = data.drop(columns=categorical_columns)

    ## Train-Test Split
    y = data[args.target]
    X = data.drop(columns=[args.target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    logger.info(
        f"Splitted input dataset shapes are: \
            X_train= {X_train.shape}, y_train= {y_train.shape},\
                X_test= {X_test.shape}, y_test= {y_test.shape}"
    )

    ## Define model parameters
    model_params = {
        "eta": args.eta,
        "max_depth": args.max_depth,
        "gamma": args.gamma,
    }

    with mlflow.start_run() as run:
        mlflow.set_tag("user_arn", args.user_arn)

        ## Initialize the XGBoost model
        model = XGBRegressor(**model_params)
        ## Train the model
        try:
            model.fit(X=X_train, y=y_train, verbose=0)
            logger.info("XGBoost model is trained.")
        except Exception as e:
            logger.exception(f"Exception {e} occured during training XGBoost model.")

        ## Calculate Regression Metrics
        try:
            regr_metrics = calc_regression_metrics(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            logger.info(
                f"Regression model preformance metrics are: {json.dumps(regr_metrics)}"
            )
        except Exception as e:
            logger.exception(
                f"Exception {e} occured during calculating regression metrics."
            )

        n_feats = [20, 50]
        plot_types = ["dot", "bar"]

        predictions = model.predict(X_test)
        signature = infer_signature(X_test, predictions)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.log_params(model_params)
        mlflow.log_metrics(regr_metrics)

        ## SHAP
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X=X_test, y=y_test, check_additivity=False)

        for n in n_feats:
            for t in plot_types:
                shap.summary_plot(
                    shap_values,
                    features=X_test,
                    feature_names=X_test.columns,
                    max_display=n,
                    plot_type=t,
                    # matplotlib=True,
                    show=False,
                )
                # plt.savefig(f"shap_{t}_{n}_xgboost.png", dpi=150, bbox_inches="tight")
                mlflow.log_artifact(f"shap_{t}_{n}_xgboost.png")
                plt.clf()
                plt.close()

        experiment = mlflow.get_experiment_by_name(args.experiment_name)

        logger.info(f"experiment_id={experiment.experiment_id}")
        logger.info(f"artifact_location={experiment.artifact_location}")
        logger.info(f"tags={experiment.tags}")
        logger.info(f"lifecycle_stage={experiment.lifecycle_stage}")
        logger.info(f"artifact_uri={mlflow.get_artifact_uri()}")
        logger.info(f"run_id={mlflow.active_run().info.run_id}")

    mlflow.end_run()
    logger.info("done")
