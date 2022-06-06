import configparser
import json
import pathlib
import logging
from utils.utils import get_logger

logger = get_logger(name=pathlib.Path(__file__))


def load_config_file(config_path: str):
    """Returns the config file loaded from the specified path, as a dictionary.

    Args:
        config_path: _description_
    """
    logger.info("Loading config file..")
    converters = {
        "list_int": lambda x: [int(i.strip()) for i in x.split(", ")],
        "list_none": lambda x: None if x.lower() == "none" else int(x),
        "list_str": lambda x: [i.strip() for i in x.split(",")],
        "pathlib": lambda x: pathlib.Path(x),
    }
    config_data = configparser.ConfigParser(converters=converters)
    config_data.read(config_path)
    config_file = {}
    # Paths config
    section = "paths"
    config_file["config_path"] = config_data.getpathlib(section, "config_path")
    config_file["data_path"] = config_data.getpathlib(section, "data_path")
    config_file["secret_names_path"] = config_data.getpathlib(
        section, "secret_names_path"
    )
    config_file["post_conv_data_path"] = config_data.getpathlib(
        section, "post_conv_data_path"
    )
    config_file["pre_conv_data_path"] = config_data.getpathlib(
        section, "pre_conv_data_path"
    )
    config_file["ma_ltv_data_path"] = config_data.getpathlib(
        section, "ma_ltv_data_path"
    )

    # preprocessing variables config
    section = "preprocessing variables"
    config_file["preprocess_data"] = config_data.getboolean(section, "preprocess_data")

    # # Unwanted Features config
    # section = "unwanted features"
    config_file["unwanted_features"] = config_data.getlist_str(
        section, "unwanted_features"
    )

    ### extra funcs
    # getboolean
    # getlist_str
    # get
    # getint
    # getlist_int

    logger.info(f"Config file at {config_path} is Loaded")

    return config_file
