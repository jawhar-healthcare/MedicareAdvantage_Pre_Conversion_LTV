import numpy as np
import pandas as pd
import boto3
import json
from pyparsing import col
from sqlalchemy import create_engine, text
import psycopg2
from typing import Optional, Tuple, Union
import logging
import pathlib
from logging import handlers
from utils.utils import get_secret, get_logger, load_data
from utils.load_config_file import load_config_file

from utils.ma_preprocessing_utils import get_post_conversion_data, get_united_features
from utils.ma_pre_conversion import get_ma_pre_conversion_data

from warnings import filterwarnings

filterwarnings("ignore")

# logging.basicConfig(level= logging.INFO)

logger = get_logger(name=pathlib.Path(__file__))
CONFIG_PATH = "config/config.ini"


def main():

    ## Load paths from config file
    config = load_config_file(config_path=CONFIG_PATH)

    preprocess_data = config["preprocess_data"]

    if preprocess_data:
        ## MA Pre-Conversion Data
        pre_data = get_ma_pre_conversion_data()
    else:
        pre_data = load_data(data_path=config["pre_conv_data_path"])

    ## MA Post-Conversion Data
    post_data = get_post_conversion_data(data_path=config["post_conv_data_path"])

    ## Merge Pre-conv MA data with Post-Conv LTV data
    ma_postconv = pd.merge(
        pre_data,
        post_data,
        left_on=["application_id", "policy_id"],
        right_on=["post_raw_application_id", "post_raw_policy_id"],
        how="left",
        suffixes=("", "_xp"),
    )
    ## Drop Duplicated Column names with "_x" suffix
    duplicated_columns = [x for x in ma_postconv.columns if "_xp" in x]
    ma_postconv = ma_postconv.drop(columns=duplicated_columns)

    ltv_feat = [feat for feat in ma_postconv.columns if "ltv" in feat.lower()][0]

    ma_postconv.rename(columns={ltv_feat: "LTV"}, inplace=True)

    ma_postconv = get_united_features(
        df=ma_postconv, features_with=config["unite_features_with"]
    )

    ma_postconv.to_csv(config["ma_ltv_data_path"], index=False)

    print(pre_data)

    # # ## Modify LTV values. Replace Blanks with Zero.
    # ma_postconv_jorn_zip_tu["mod_LTV"] = (
    #     ma_postconv_jorn_zip_tu["post_raw_LTV"] / 1.95
    # ).fillna(0)


if __name__ == "__main__":
    main()
