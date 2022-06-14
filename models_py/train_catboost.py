from ast import arg
import logging
import numpy as np
import pandas as pd
import pathlib
from catboost import CatBoostRegressor

import shap
import json
import argparse

import mlflow

# import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

# import seaborn as sns
import matplotlib.pyplot as plt

# from utils.load_config_file import load_config_file
from utils.utils import calc_regression_metrics
from warnings import filterwarnings

filterwarnings("ignore")

# logger = get_logger(name=pathlib.Path("train_catboost.py"))
logging.basicConfig(level=logging.INFO)


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
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--loss_function", type=str, default="RMSE")
    parser.add_argument("--target", type=str, default="LTV")
    # parser.add_argument(
    #     "--data_path", type=str, default="data/with_zcta/ma_ltv_merged.csv"
    # )
    parser.add_argument(
        "--train",
        type=str,
        default="s3://hc-data-science/pre-conversion-ma-ltv/data/post-processed/ma_ltv_train.csv",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="s3://hc-data-science/pre-conversion-ma-ltv/data/post-processed/ma_ltv_test.csv",
    )

    args, _ = parser.parse_known_args()

    logging.info(f"Tracking URL: {args.tracking_uri}")
    logging.info(f"Experiment Name: {args.experiment_name}")

    ## Initialize MLflow params
    try:
        mlflow.create_experiment(args.experiment_name, "s3://hc-prd-mlflow-bucket")
        logging.info(f"Experiment {args.experiment_name} is successfully created.")
    except MlflowException as ex:
        logging.exception(f"Error creating experiment {ex}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    ## Load the Dataset
    train_data = pd.read_csv(args.train, na_filter=False, low_memory=False)
    test_data = pd.read_csv(args.test, na_filter=False, low_memory=False)

    # determine categorical and numerical features
    numerical_cols = list(
        train_data.select_dtypes(include=["int64", "float64"]).columns
    )
    # numerical_cols.remove(args.target)

    categorical_cols = list(
        train_data.select_dtypes(include=["object", "bool"]).columns
    )

    for col in numerical_cols:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)
    for col in categorical_cols:
        train_data[col] = train_data[col].fillna("N/A")
        test_data[col] = test_data[col].fillna("N/A")

    ## Split Predictors and response variables
    y_train = train_data[args.target]
    y_test = test_data[args.target]
    X_train = train_data.drop(columns=[args.target])
    X_test = test_data.drop(columns=[args.target])

    # config = load_config_file(config_path=CONFIG_PATH)

    ## Define model parameters
    model_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "loss_function": args.loss_function,
    }
    # logging.info(categorical_cols)

    with mlflow.start_run() as run:
        mlflow.set_tag("user_arn", args.user_arn)

        ## Initialize the CatBoost model
        model = CatBoostRegressor(**model_params)
        ## Train the model
        try:
            model.fit(X=X_train, y=y_train, cat_features=categorical_cols, verbose=0)
            logging.info("CatBoost model is trained.")
        except Exception as e:
            logging.exception(f"Exception {e} occured during training CatBoost model.")

        ## Calculate Regression Metrics
        try:
            regr_metrics = calc_regression_metrics(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            logging.info(
                f"Regression model preformance metrics are: {json.dumps(regr_metrics)}"
            )
        except Exception as e:
            logging.exception(
                f"Exception {e} occured during calculating regression metrics."
            )
        # n_feats = [20, 50]
        # plot_types = ["dot", "bar"]

        predictions = model.predict(X_test)
        signature = infer_signature(X_test, predictions)
        ## Log Parameters
        mlflow.log_params(model_params)
        ## Log Metrics
        for key in regr_metrics:
            logging.info(f"{key}: {regr_metrics[key]}")
            mlflow.log_metric(f"{key}", regr_metrics[key])

        ## SHAP
        # shap.initjs()
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X_test)

        # for n in n_feats:
        #     for t in plot_types:
        #         shap.summary_plot(
        #             shap_values,
        #             features=X_test,
        #             feature_names=X_test.columns,
        #             max_display=n,
        #             plot_type=t,
        #             # matplotlib=True,
        #             show=False,
        #         )
        #         plt.savefig(f"shap_{t}_{n}_catboost.png", dpi=150, bbox_inches="tight")
        #         mlflow.log_artifact(f"shap_{t}_{n}_catboost.png")
        #         plt.clf()
        #         plt.close()

        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        logging.info("Pre-Conversion MA LTV Catboost Model is saved in MLfLow.")
        # logging.info(f"experiment_id={experiment.experiment_id}")
        # logging.info(f"artifact_location={experiment.artifact_location}")
        # logging.info(f"tags={experiment.tags}")
        # logging.info(f"lifecycle_stage={experiment.lifecycle_stage}")
        # logging.info(f"artifact_uri={mlflow.get_artifact_uri()}")
        # logging.info(f"run_id={mlflow.active_run().info.run_id}")

    # mlflow.end_run()
    # logging.info("done")
