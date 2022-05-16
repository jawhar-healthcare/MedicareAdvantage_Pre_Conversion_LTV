import numpy as np 
import pandas as pd 
import boto3
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2

from utils.load_MA_data import get_MA_data
from utils.load_journaya_data import get_jornaya_data
from utils.load_zip_enrch_data import get_zip_enrch_data
from utils.preprocess_transunion import preprocess_transunion_data

from warnings import filterwarnings
filterwarnings("ignore")

## AWS RedShift Credentials
REDSHIFT_USERNAME = ""
REDSHIFT_PASSWORD = ""
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

    ma_data['first_name'] = ma_data["owner_name"].apply(lambda x: x.split(" ")[0])
    ma_data['last_name'] = ma_data["owner_name"].apply(lambda x: x.split(" ")[0])

    ma_data.drop(columns=['owner_name'], inplace= True)

    ## Load MA Post-Conversion LTV Model Predictions data
    post_conv_data = pd.read_csv("data/MA LTV model predictionsFINAL.csv", low_memory=False)

    ## Convert LTV value Strings to Floats
    post_conv_data["LTV"] = post_conv_data["LTV"].str.replace('$', '').str.replace(',', '').astype(float)

    ## Keep only relevant features from MA Post Conversion LTV Data
    relevant_features = [
        'application_id', 
        'policy_id', 
        'medicare_number',
        'cancellation_model_prediction',
        'probability_of_cancellation',
        'duration_model_prediction',
        'LTV', 
        'coverage_duration'
    ]
    irrelevant_features = [feat for feat in post_conv_data.columns if feat not in relevant_features]

    ## Keep only relevant columns
    post_conv_data = post_conv_data.drop(columns=irrelevant_features)

    ## Add a prefix "post_raw_" to all columns
    post_conv_data = post_conv_data.add_prefix("post_raw_")

    ## Merge MA data with Post-Conv LTV data
    ma_postconv = pd.merge(
        ma_data, 
        post_conv_data, 
        left_on= ['application_id', 'policy_id'],
        right_on= ['post_raw_application_id', 'post_raw_policy_id'],
        how= 'left',
        suffixes= ("", "_x")
    )
    ## Drop Duplicated Column names with "_x" suffix
    duplicated_columns = [x for x in ma_postconv.columns if "_x" in x]
    ma_postconv = ma_postconv.drop(columns=duplicated_columns)

    # ## Rename "Name" features
    # ma_postconv.rename(
    #     columns= {
    #         "post_raw_owner_first_name": "owner_first_name", 
    #         "post_raw_owner_last_name": "owner_last_name",
    #         "post_raw_owner_phone": "owner_phone_trAASDSHSD"
    #     },
    #     inplace= True
    # )

    ## List all unique Lead IDs from MA_POSTCONV dataset
    lead_ids = tuple(
        set(
            ma_postconv[ma_postconv["lead_id"] \
            .notna()]["lead_id"] \
            .astype('Int64')
        )
    )

    ## Get Jornaya data
    jrn_data = get_jornaya_data(leads= lead_ids, engine= hc_engine)

    ## Merge Jornaya data with MA_POSTCONV data
    ma_postconv_jorn = pd.merge(
        ma_postconv, 
        jrn_data, 
        left_on= ['lead_id'],
        right_on= ['jrn_boberdoo_lead_id'],
        how= 'left',
        suffixes= ("", "_j")
    )

    ## Drop Duplicated Column names with "_j" suffix
    duplicated_columns = [x for x in ma_postconv_jorn.columns if "_j" in x]
    duplicated_columns.append('jrn_boberdoo_lead_id')
    ma_postconv_jorn = ma_postconv_jorn.drop(columns=duplicated_columns)

    ## Get Zipcode Enriched Data
    zip_data = get_zip_enrch_data(engine= hc_engine)

    ## Merge Zipcode data with MA_POSTCONV_JORNAYA data
    ma_postconv_jorn_zip = pd.merge(
        ma_postconv_jorn, 
        zip_data, 
        left_on= ['app_zip_code'],
        right_on= ['zip_zipcode'],
        how= 'left',
        suffixes= ("", "_z")
    )

    ## Drop Duplicated Column names with "_z" suffix
    duplicated_columns = [x for x in ma_postconv_jorn_zip.columns if "_z" in x]
    duplicated_columns.append('zip_zipcode')
    ma_postconv_jorn_zip = ma_postconv_jorn_zip.drop(columns=duplicated_columns)

    ## Some preprocessing on merged dataset
    ma_postconv_jorn_zip['first_name'] = ma_postconv_jorn_zip["first_name"].str.lower()
    ma_postconv_jorn_zip['last_name'] = ma_postconv_jorn_zip["last_name"].str.lower()

    ma_phone_nums = list(ma_postconv_jorn_zip["owner_phone"].unique())
    ma_first_names = list(ma_postconv_jorn_zip['first_name'].unique())
    ma_last_names = list(ma_postconv_jorn_zip['last_name'].unique())


    ## Get Transunion Data
    tu_data = preprocess_transunion_data(
        query_string= '''select * from PROD_STAGE.WEB_TRACKING.INTERNAL_TRANSUNION_EVENT''',
        username= '', 
        password= '',
        account ='uza72979.us-east-1',
        phone_numbers_list= ma_phone_nums,
        first_names_list= ma_first_names,
        last_names_list= ma_last_names 
    )

    ## Merge Trandunion data with MA_POSTCONV_JORNAYA_ZIP data
    ma_postconv_jorn_zip_tu = pd.merge(
        ma_postconv_jorn_zip, 
        tu_data, 
        left_on= ['owner_phone', "first_name", "last_name"],
        right_on= ['tu_PHONE_NUMBER', "tu_FIRST_NAME", "tu_LAST_NAME"],
        how= 'left',
        suffixes= ("", "_tu")
    )

    ## Drop Duplicated Column names with "_tu" suffix
    duplicated_columns = [x for x in ma_postconv_jorn_zip_tu.columns if "_tu" in x]
    duplicated_columns.append('tu_PHONE_NUMBER')
    duplicated_columns.append("tu_FIRST_NAME")
    duplicated_columns.append("tu_LAST_NAME")
    ma_postconv_jorn_zip_tu = ma_postconv_jorn_zip_tu.drop(columns=duplicated_columns)

    ## Save to file
    ma_postconv_jorn_zip_tu.to_csv("data/ma_postconv_jorn_zip_tu.csv", index= False)

    print(duplicated_columns)





if __name__ == "__main__":
    main()
