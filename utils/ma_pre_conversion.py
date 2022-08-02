import os
import secrets
import numpy as np
import pandas as pd
import pathlib
import boto3
import json
from sqlalchemy import create_engine, text
import psycopg2
from typing import Optional, Tuple, Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
)

from utils.utils import get_logger, get_secret
from utils.load_config_file import load_config_file
from utils.ma_preprocessing_utils import (
    get_MedAdv_data,
    get_jornaya_data,
    get_zip_data,
    get_county_city_data,
    preprocess_transunion_data,
)
from dotenv import load_dotenv

logger = get_logger(name=pathlib.Path(__file__))
PATHS_CONFIG = "config/config.ini"


def get_aws_credentials(secrets_path: Union[str, pathlib.Path], db_list: list):
    """
    Download AWS credentials from AWS Secretsmanager

    Args:
        secrets_path: Path to the secrets.json file
        db_list: List of databases to extract data from.

    Returns:
        AWS Credentials
    """
    ## Check if system is configured with AWS IAM account
    try:
        user_arn = boto3.resource("iam").CurrentUser().arn
        logger.info(f"User Amazon Resource Name (ARN) is {user_arn}")
    except:
        logger.exception(
            "System is not configured with AWS IAM account; ARN is not available."
        )
    aws_secrets = {}
    try:
        with open(secrets_path) as f:
            secret_names = json.load(f)  # Load Secret Names
        for db in db_list:
            db_u, db_l = db.upper(), db.lower()
            secret = json.loads(
                get_secret(secret_name=secret_names[db_u])["SecretString"]
            )
            aws_secrets.update({db_l: secret})
        logger.info("AWS Redshift Credentials are loaded.")
    except:
        logger.exception("Could not load AWS Redshift Credentials. ")

    return aws_secrets


def get_aws_engines(secrets_path: Union[str, pathlib.Path], db_list: list):
    """
    Load AWS Redshift engines

    Args:
        secrets_path:Path to the secrets.json file
        db_list: List of databases to extract data from.

    Returns:
        list: AWS engines
    """
    aws_secrets = get_aws_credentials(secrets_path=secrets_path, db_list=db_list)
    aws_engines = {}
    for db in db_list:
        db = db.lower()
        ## Create AWS Redshift engine
        engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" % (
            aws_secrets[db]["username"],
            aws_secrets[db]["password"],
            aws_secrets[db]["host"],
            aws_secrets[db]["port"],
            str(db),
        )
        aws_engines[db] = create_engine(engine_string)
    return aws_engines


def get_ma_pre_conversion_data():
    """
    Preprocessed the data from various sources and merges/concatenates
    them into a singular "pre-conversion MA LTV" dataset to be used for
    model training.

    Returns:
        Pre-conversion MA LTV dataset
    """

    ## Load paths from config file
    paths = load_config_file(config_path=PATHS_CONFIG)

    ## Load .env file variables
    load_dotenv(dotenv_path=paths["env_file_path"])

    # Get AWS Engines
    database_list = ["hc", "isc"]
    aws_engines = get_aws_engines(
        secrets_path=paths["secret_names_path"], db_list=database_list
    )

    ### Medicare Advantage Data ###

    ## Load Medicare Advantage Data
    ma_data = get_MedAdv_data(engine=aws_engines["isc"], save_csv=True)
    data = ma_data.copy()

    ## List all unique Lead IDs from MA_POSTCONV dataset
    lead_ids = tuple(set(data[data["lead_id"].notna()]["lead_id"].astype("Int64")))

    ### Jornaya data ###

    ## Get Jornaya data
    jrn_data = get_jornaya_data(leads=lead_ids, engine=aws_engines["hc"], save_csv=True)

    ## Merge Jornaya data with MA_POSTCONV data
    data = pd.merge(
        data,
        jrn_data,
        left_on=["lead_id"],
        right_on=["jrn_lead_id"],
        how="left",
        suffixes=("", "_xj"),
    )

    ## Drop Duplicated Column names with "_xj" suffix
    duplicated_columns = [x for x in data.columns if "_xj" in x]
    duplicated_columns.append("jrn_lead_id")
    data = data.drop(columns=duplicated_columns)

    ### Zipcode Data ###

    ## Get Zipcode ZCTA Data
    zip_data = get_zip_data(
        username=os.getenv("snowflake_username"),
        password=os.getenv("snowflake_password"),
        account=os.getenv("snowflake_account"),
        save_csv=True,
    )

    zip_data = zip_data.groupby(by="zcta_zcta").first().reset_index()

    ## Merge Zipcode data with MA_POSTCONV_JORNAYA data
    data = pd.merge(
        data,
        zip_data,
        left_on=["app_zip_code"],
        right_on=["zcta_zcta"],
        how="left",
        suffixes=("", "_xz"),
    )

    ## Drop Duplicated Column names with "_z" suffix
    duplicated_columns = [x for x in data.columns if "_xz" in x]
    duplicated_columns.append("zcta_zcta")
    data = data.drop(columns=duplicated_columns)

    # ## GET COUNTY-CITY DATA
    cc_data = get_county_city_data(
        username=os.getenv("snowflake_username"),
        password=os.getenv("snowflake_password"),
        account=os.getenv("snowflake_account"),
    )

    cc_data = cc_data.groupby(by="zcta").first().reset_index()

    data = pd.merge(
        data,
        cc_data,
        left_on=["app_zip_code"],
        right_on=["zcta"],
        how="left",
        suffixes=("", "_xcc"),
    )
    ## Drop Duplicated Column names with "_xcc" suffix
    duplicated_columns = [x for x in data.columns if "_xcc" in x]
    duplicated_columns.append("zcta")
    data = data.drop(columns=duplicated_columns)

    # ## Some preprocessing on merged dataset
    data["first_name"] = data["first_name"].str.lower()
    data["last_name"] = data["last_name"].str.lower()

    ma_phone_nums = list(data["owner_phone"].unique())
    ma_first_names = list(data["first_name"].unique())
    ma_last_names = list(data["last_name"].unique())

    ### Transunion Data ###

    ## Get Transunion Data
    tu_data = preprocess_transunion_data(
        username=os.getenv("snowflake_username"),
        password=os.getenv("snowflake_password"),
        account=os.getenv("snowflake_account"),
        phone_numbers_list=ma_phone_nums,
        first_names_list=ma_first_names,
        last_names_list=ma_last_names,
        save_csv=True,
    )

    tu_data = (
        tu_data.groupby(by=["tu_phone_number", "tu_first_name", "tu_last_name"])
        .first()
        .reset_index()
    )

    ## Merge Trandunion data with MA_POSTCONV_JORNAYA_ZIP data
    data = pd.merge(
        data,
        tu_data,
        left_on=["owner_phone", "first_name", "last_name"],
        right_on=["tu_phone_number", "tu_first_name", "tu_last_name"],
        how="left",
        suffixes=("", "_xtu"),
    )

    ## Drop Duplicated Column names with "_tu" suffix
    duplicated_columns = [x for x in data.columns if "_xtu" in x]
    duplicated_columns.append("tu_phone_number")
    duplicated_columns.append("tu_first_name")
    duplicated_columns.append("tu_last_name")
    data = data.drop(columns=duplicated_columns)

    data = data.drop_duplicates(keep="last", ignore_index=True)

    data.to_csv(paths["pre_conv_data_path"], index=False)

    return data
