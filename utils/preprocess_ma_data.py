import imp
from unicodedata import category
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    LabelEncoder, 
    OneHotEncoder
)
from sklearn.pipeline import Pipeline, make_pipeline

def preprocess_ma_data(
    data_path: str
):

    ## Load Pre-Conversion MA data with Post-Conversion LTV Values
    ma_data = pd.read_csv(data_path, low_memory=False)

    ## Remove unwanted features
    unwanted_features = [
        "application_id",
        "owner_email",
        "policy_id",
        "owner_id",
        "owner_phone",
        "pol_zip_code",
        "parent_application_id",
        "bk_product_type",
        "lead_id",
        "first_name",
        "last_name",
        "jrn_error",
        "tu_GROUP_ID"
    ]

    ## Remove any post-conversion data features
    unwanted_features =  unwanted_features + [p for p in ma_data.columns if "post_raw" in p.lower()]

    ma_data.drop(columns= unwanted_features, inplace= True)

    ## Get all the numeric features
    numeric_ma_data = ma_data.select_dtypes(include= "number")

    ## Get all the categorical data
    category_ma_data = ma_data.select_dtypes(include= "object" or "category")


    ## Label encoder for categorical columns data

    cate_transform = ColumnTransformer([
        ('cate_label_enc', LabelEncoder(), [1,6])
    ], remainder='passthrough')






    print(ma_data.shape)
