from ast import arg
import logging
import numpy as np
import pandas as pd
import pathlib
from catboost import CatBoostRegressor

import shap
import json
import argparse
import pickle
import mlflow

# import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

# import seaborn as sns
import matplotlib.pyplot as plt

# from utils.load_config_file import load_config_file
from utils.utils import calc_regression_metrics

# from utils.load_config_file import load_config_file
from warnings import filterwarnings

filterwarnings("ignore")

# logger = get_logger(name=pathlib.Path("train_catboost.py"))
logging.basicConfig(level=logging.INFO)


CONFIG_PATH = "config/config.ini"


def get_numeric_categorical(df: pd.DataFrame, target: str):
    # config = load_config_file(config_path=CONFIG_PATH)

    # determine categorical and numerical features
    numerical_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)

    categorical_cols = list(
        df.select_dtypes(include=["object", "string", "bool"]).columns
    )

    force_categorical = ["zip"]
    for cat in force_categorical:
        feats = [col for col in df.columns if cat.lower() in col.lower()]

    for f in feats:
        if f in numerical_cols:
            numerical_cols.remove(f)
        if f not in categorical_cols:
            categorical_cols.append(f)

    # print("cate:", categorical_cols)
    # print("num:", len(numerical_cols))

    predictors = list(df.columns)
    predictors.remove(target)

    ## Save training Feature names
    features = {
        "numeric": numerical_cols,
        "categorical": categorical_cols,
        "predictor": predictors,
        "response": target,
    }
    with open("features.pkl", "wb+") as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact("features.pkl")

    return numerical_cols, categorical_cols


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

    parser.add_argument(
        "--train",
        type=str,
        default="s3://hc-data-science/pre-conversion-ma-ltv/data/post-processed/train_tu_jrn_null.csv",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="s3://hc-data-science/pre-conversion-ma-ltv/data/post-processed/test_tu_jrn_null.csv",
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
    train_data = pd.read_csv(args.train, low_memory=False)
    test_data = pd.read_csv(args.test, low_memory=False)

    ## Some preprocessing and get num and cat columns
    num_cols, cat_cols = get_numeric_categorical(df=train_data, target=args.target)

    for col in num_cols:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna(0)
    for col in cat_cols:
        train_data[col] = train_data[col].fillna("N/A")
        test_data[col] = test_data[col].fillna("N/A")

    num_cols.remove(args.target)

    ## Split Predictors and response variables
    y_train = train_data[args.target]
    y_test = test_data[args.target]
    X_train = train_data.drop(columns=[args.target])
    X_test = test_data.drop(columns=[args.target])

    for cate in cat_cols:
        X_train[cate] = X_train[cate].astype(str)
        X_test[cate] = X_train[cate].astype(str)

    ## Define model parameters
    model_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "loss_function": args.loss_function,
    }

    # with mlflow.start_run() as run:
    mlflow.set_tag("user_arn", args.user_arn)

    ## Initialize the CatBoost model
    model = CatBoostRegressor(**model_params)
    ## Train the model
    try:
        model.fit(X=X_train, y=y_train, cat_features=cat_cols, verbose=0)
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
    n_feats = [50]
    plot_types = ["dot"]

    try:
        predictions = model.predict(X_test)
    except Exception as e:
        print(e)

    signature = infer_signature(X_test, predictions)
    ## Log Parameters
    mlflow.log_params(model_params)
    ## Log Metrics
    for key in regr_metrics:
        logging.info(f"{key}: {regr_metrics[key]}")
        mlflow.log_metric(f"{key}", regr_metrics[key])

    ## SHAP
    shap.initjs()
    explainer = shap.TreeExplainer(model)

    ## For train
    shap_values = explainer.shap_values(X_train)
    for n in n_feats:
        for t in plot_types:
            shap.summary_plot(
                shap_values,
                features=X_train,
                feature_names=X_train.columns,
                max_display=n,
                plot_type=t,
                # matplotlib=True,
                show=False,
            )
            plt.savefig(f"shap_{n}_train.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact(f"shap_{n}_train.png")
            plt.clf()
            plt.close()

    del shap_values

    ## For test
    shap_values = explainer.shap_values(X_test)
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
            plt.savefig(f"shap_{n}_test.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact(f"shap_{n}_test.png")
            plt.clf()
            plt.close()
    del shap_values

    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    mlflow.sklearn.log_model(model, "model", signature=signature)

    logging.info("Pre-Conversion MA LTV Catboost Model is saved in MLfLow.")
    logging.info(f"experiment_id={experiment.experiment_id}")
    logging.info(f"artifact_location={experiment.artifact_location}")
    logging.info(f"tags={experiment.tags}")
    logging.info(f"lifecycle_stage={experiment.lifecycle_stage}")
    logging.info(f"artifact_uri={mlflow.get_artifact_uri()}")
    logging.info(f"run_id={mlflow.active_run().info.run_id}")

    # mlflow.end_run()
    logging.info("done")
