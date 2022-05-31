
# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:   
# https://aws.amazon.com/developers/getting-started/python/
import sys
import pathlib 
import boto3
import base64
from botocore.exceptions import ClientError
import logging
from logging import handlers

from models_py.catboost import X

### AWS Secrets Manager retrieval code

def get_secret(secret_name: str,):
    # Create a Secrets Manager client
    session = boto3.session.Session()

    client = session.client(
        service_name= 'secretsmanager',
        region_name= session.region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response['SecretBinary']
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


## Calculation of performance metrics
def calc_regression_metrics(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):
    import numpy as np
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    ## Predictions
    train_preds = model.predict(X= X_train)
    test_preds = model.predict(X= X_test)

    ## MSE
    train_mse = mean_squared_error(
        y_true= y_train,
        y_pred= train_preds
    )
    test_mse = mean_squared_error(
        y_true= y_test,
        y_pred= test_preds
    )

    ## MAE
    train_mae = mean_absolute_error(
        y_true= y_train,
        y_pred= train_preds
    )
    test_mae = mean_absolute_error(
        y_true= y_test,
        y_pred= test_preds
    )

    ## R2 Score
    train_r2s = r2_score(
        y_true= y_train,
        y_pred= train_preds
    )
    test_r2s = r2_score(
        y_true= y_test,
        y_pred= test_preds
    )

    regression_metrics = {
        "train_preds": train_preds,
        "test_preds": test_preds,
        "test_preds_mean": np.mean(test_preds),
        "MAE_train": train_mae,
        "MAE_test": test_mae,
        "RMSE_train": np.sqrt(train_mse),
        "RMSE_test": np.sqrt(test_mse),
        # "MSE_train": train_mse,
        # "MSE_test": test_mse,
        "R2_score_train": train_r2s,
        "R2_score_test": test_r2s
    }

    return regression_metrics




