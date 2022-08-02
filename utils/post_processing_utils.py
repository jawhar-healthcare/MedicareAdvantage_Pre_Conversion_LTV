import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from utils.load_config_file import load_config_file


def process_catboost(
    data: pd.DataFrame, config_path: str, for_training=True, save_csv=True
):
    """
    Preprocess the input dataframe for data exploration, training or
    evaluation of CatBoost model.
    If training is to be done, dataframe can be standardized/normalized.
    If training is not to be done, original data values are retained.

    Args:
        data: Input Dataframe
        config_path: Path to config file
        for_training: If dataset is to be used for training. Defaults to True.
        save_csv: Save as a CSV. Defaults to True.

    Returns:
        X_train: Train dataset with response variable (LTV)
        X_test: Test dataset with response variable (LTV)
    """

    config = load_config_file(config_path=config_path)
    target = config["target"]

    ## Remove unwanted features
    unwanted_features = config["unwanted_features"]
    # print(unwanted_features)
    # unwanted_features = [f for f in unwanted_features if f is not "Null"]
    ## Remove any post-conversion data features; replace nulls with 0 or N/A
    unwanted_features = (
        unwanted_features
        + [p for p in data.columns if "post_raw" in p.lower()]
        # + [c for c in data.columns if len(data[c].fillna(0).unique()) == 1]
    )
    data = data.drop(columns=unwanted_features)

    # determine categorical and numerical features
    numerical_ix = list(data.select_dtypes(include=["int64", "float64"]).columns)
    numerical_ix.remove(target)

    categorical_ix = list(
        data.select_dtypes(include=["object", "string", "bool"]).columns
    )

    ## Initialize the Scaler
    if config["normalize_type"].lower() == "standardize":
        scaler = StandardScaler()
    elif config["normalize_type"].lower() == "normalize":
        scaler = MinMaxScaler()
    else:
        scaler = None

    ## Initialize the transformer
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categoric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="N/A"),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_ix),
            ("cat", categoric_transformer, categorical_ix),
        ]
    )

    ## Predictor-Response Split
    y = data[target]
    X = data.drop(columns=[target])

    if for_training:
        ## Transform Predictor data
        X = transformer.fit_transform(X)
        X = pd.DataFrame(data=X, columns=numerical_ix + categorical_ix)

    ## Train-Test Split
    test_size = float(config["train_test_ratio"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_train[target] = y_train
    X_test[target] = y_test

    if save_csv:
        X_train.to_csv(config["train_data_path"], index=False)
        X_test.to_csv(config["test_data_path"], index=False)

    return X_train, X_test
