# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/
from asyncio.log import logger
import sys
import pathlib
import boto3
import base64
from botocore.exceptions import ClientError
import logging
from logging import handlers
from typing import Union

from pyparsing import col

# from models_py.catboost import c

### AWS Secrets Manager retrieval code


def get_secret(
    secret_name: str,
):
    # Create a Secrets Manager client
    session = boto3.session.Session()

    client = session.client(
        service_name="secretsmanager", region_name=session.region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
        else:
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response["SecretBinary"]
            )

    return get_secret_value_response


### Logger func


def get_logger(name):

    formatter = logging.Formatter(
        "%(asctime)s — %(filename)s - %(lineno)d — %(levelname)s — %(message)s"
    )
    file_name = f"{name.stem}.log"

    logger = logging.getLogger(str(name))
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename=f"logs/{file_name}")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def load_config_dict(config_path: str) -> dict:
    """
    Loads the config.ini file from the specified path, as a dictionary.
    Args:
        path_to_config_file: String containing path to Config file.
    Returns:
        dict: The config file in a dict format.
    """
    from configparser import ConfigParser

    config = ConfigParser()
    config.optionxform = str
    config.read(config_path)
    config_items = config.sections()
    if "DEFAULT" in config_items:
        config_items.remove("DEFAULT")

    config_dict = {}
    for item in config_items:
        config_dict.update({item: {}})
        for parameter, value in config[item].items():
            config_dict[item].update({parameter: value})

    return config_dict


def load_data(data_path: Union[str, pathlib.Path]):
    import pandas as pd

    logger = get_logger(name=pathlib.Path(__file__))
    try:
        if str(data_path).endswith(".csv"):
            data = pd.read_csv(str(data_path), low_memory=False)
        elif str(data_path).endswith(".xlsx"):
            data = pd.read_excel(str(data_path))
        elif str(data_path).endswith(".parquet"):
            data = pd.read_parquet(str(data_path))
        else:
            data = pd.DataFrame(data=[], columns=["LTV"])
            logger.info(f"Unrecognized file type at {data_path}")
    except Exception as e:
        logger.exception(f"Exception {e} occured while loading file at {data_path}")

    return data


## Calculation of performance metrics
def calc_regression_metrics(model, X_train, y_train, X_test, y_test):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )

    ## Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    if train_preds.shape[1] > 1:
        train_preds = train_preds[:, 0]
        tr_pred_stdev = pd.DataFrame(train_preds).apply(lambda x: np.sqrt(x)).mean()[0]
    else:
        tr_pred_stdev = None

    if test_preds.shape[1] > 1:
        test_preds = test_preds[:, 0]
        ts_pred_stdev = pd.DataFrame(test_preds).apply(lambda x: np.sqrt(x)).mean()[0]
    else:
        ts_pred_stdev = None

    ## MSE
    train_mse = mean_squared_error(y_true=y_train, y_pred=train_preds)
    test_mse = mean_squared_error(y_true=y_test, y_pred=test_preds)

    ## MAE
    train_mae = mean_absolute_error(y_true=y_train, y_pred=train_preds)
    test_mae = mean_absolute_error(y_true=y_test, y_pred=test_preds)

    ## R2 Score
    train_r2s = r2_score(y_true=y_train, y_pred=train_preds)
    test_r2s = r2_score(y_true=y_test, y_pred=test_preds)

    regression_metrics = {
        "MAE_train": train_mae,
        "MAE_test": test_mae,
        "RMSE_train": np.sqrt(train_mse),
        "RMSE_test": np.sqrt(test_mse),
        "R2_score_train": train_r2s,
        "R2_score_test": test_r2s,
        "train_preds_mean_stdev": tr_pred_stdev,
        "test_preds_mean_stdev": ts_pred_stdev,
        "test_preds_mean": np.mean(test_preds),
    }

    return regression_metrics
