import numpy as np 
import pandas as pd 
import boto3
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2

from utils.load_MA_data import get_MA_data
from utils.load_journaya_data import get_jornaya_data

from warnings import filterwarnings
filterwarnings("ignore")

## AWS RedShift Credentials
REDSHIFT_USERNAME = "rutvik_bhende"
REDSHIFT_PASSWORD = "B08tLOo-n69d5t4"
PORT = 5439

def main():
    ## ISC creds
    redshift_endpoint_isc = "isc-prd-data-warehouse.c3aww65gl0dd.us-east-1.redshift.amazonaws.com"
    dbname_isc = "isc"
    ## HC creds
    redshift_endpoint_hc = "data-warehouse.aws.healthcare.com"
    dbname_hc = "hc"
    ## PH creds
    redshift_endpoint_ph = "data-warehouse.pivothealth.com"
    dbname_ph = "ph"

    ## Create AWS Redshift engines
    ## ISC
    isc_engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" \
                    % (REDSHIFT_USERNAME, REDSHIFT_PASSWORD, redshift_endpoint_isc, PORT, dbname_isc)
    isc_engine = create_engine(isc_engine_string)

    ## HC
    hc_engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" \
                    % (REDSHIFT_USERNAME, REDSHIFT_PASSWORD, redshift_endpoint_hc, PORT, dbname_hc)
    hc_engine = create_engine(hc_engine_string)


    ## Load Medicare Advantage Data
    ma_data = get_MA_data(engine= isc_engine)

    ## Load MA Post-Conversion LTV Model Predictions data
    post_conv_data = pd.read_csv("data/MA LTV model predictionsFINAL.csv", low_memory=False)

    ## Convert LTV value Strings to Floats
    post_conv_data["LTV"] = post_conv_data["LTV"].str.replace('$', '').str.replace(',', '').astype(float)

    ## Keep only relevant features from MA Post Conversion LTV Data
    relevant_features = [
        'application_id', 
        'medicare_number',
        'sk_effective_date',
        'cancellation_model_prediction',
        'probability_of_cancellation',
        'duration_model_prediction',
        'LTV', 
        'owner_first_name',
        'owner_last_name',
        'owner_phone',
        'coverage_duration', 
        'policy_id', 
        'sk_carrier', 
        'churn'
    ]
    irrelevant_features = [feat for feat in post_conv_data.columns if feat not in relevant_features]

    ## Keep only relevant columns
    post_conv_data = post_conv_data.drop(columns=irrelevant_features)

    ## Merge MA data with Post-Conv LTV data
    ma_postconv = pd.merge(
        ma_data, 
        post_conv_data, 
        on= ['application_id', 'policy_id'],
        how= 'left',
        suffixes= ("", "_x")
    )
    ## Drop Duplicated Column names with "_x" suffix
    duplicated_columns = [x for x in ma_postconv.columns if "_x" in x]
    ma_postconv = ma_postconv.drop(columns=duplicated_columns)


    ## List all unique Lead IDs from MA_POSTCONV dataset
    lead_ids = tuple(
        set(
            ma_postconv[ma_postconv["lead_id"] \
            .notna()]["lead_id"] \
            .astype('Int64')
        )
    )

    ## Get Jornaya data
    jrn_data = get_jornaya_data(leads= lead_ids, engine= hc_engine_string)

    ## Merge Jornaya data with MA_POSTCONV data
    ma_postconv_jorn = pd.merge(
        ma_postconv, 
        jrn_data, 
        left_on= ['lead_id'],
        right_on= ['boberdoo_lead_id'],
        how= 'left',
        suffixes= ("", "_j")
    )

    ## Drop Duplicated Column names with "_j" suffix
    duplicated_columns = [x for x in ma_postconv_jorn.columns if "_j" in x]
    duplicated_columns.append('boberdoo_lead_id')
    ma_postconv_jorn = ma_postconv_jorn.drop(columns=duplicated_columns)




