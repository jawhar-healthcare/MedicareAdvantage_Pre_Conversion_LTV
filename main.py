import numpy as np 
import pandas as pd 
import boto3
from pyparsing import col
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2
from typing import Optional, Tuple, Union

from utils.utils import (
    get_MA_data,
    preprocess_post_conversion_ltv_data,
    get_jornaya_data, 
    get_zip_enrch_data, 
    preprocess_transunion_data
)
from utils.preprocess_ma_data import preprocess_ma_data

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
    
    ### Medicare Advantage Data ###

    ## Load Medicare Advantage Data
    # ma_data = get_MA_data(engine= isc_engine)

    # ## Drop Duplicates, Keep relevant data
    # # ma_data = fill_and_drop_duplicates(df= ma_data, identifiers= ["application_id"])

    # ########

    # ### Post-Conversion LTV Data ###

    # post_conv_data = preprocess_post_conversion_ltv_data(
    #     data_path= "data/MA LTV model predictionsFINAL.csv"
    # )

    # ## Merge MA data with Post-Conv LTV data
    # ma_postconv = pd.merge(
    #     ma_data, 
    #     post_conv_data, 
    #     left_on= ['application_id', 'policy_id'],
    #     right_on= ['post_raw_application_id', 'post_raw_policy_id'],
    #     how= 'left',
    #     suffixes= ("", "_xp")
    # )
    # ## Drop Duplicated Column names with "_x" suffix
    # duplicated_columns = [x for x in ma_postconv.columns if "_xp" in x]
    # ma_postconv = ma_postconv.drop(columns=duplicated_columns)

    # ## List all unique Lead IDs from MA_POSTCONV dataset
    # lead_ids = tuple(
    #     set(
    #         ma_postconv[ma_postconv["lead_id"] \
    #         .notna()]["lead_id"] \
    #         .astype('Int64')
    #     )
    # )
    
    # ########

    # ### Jornaya data ###

    # ## Get Jornaya data
    # jrn_data = get_jornaya_data(leads= lead_ids, engine= hc_engine)

    # # jrn_data = fill_and_drop_duplicates(
    # #     df= jrn_data, 
    # #     identifiers=["jrn_boberdoo_lead_id"]
    # # )

    # ## Merge Jornaya data with MA_POSTCONV data
    # ma_postconv_jorn = pd.merge(
    #     ma_postconv, 
    #     jrn_data, 
    #     left_on= ['lead_id'],
    #     right_on= ['jrn_boberdoo_lead_id'],
    #     how= 'left',
    #     suffixes= ("", "_xj")
    # )

    # ## Drop Duplicated Column names with "_j" suffix
    # duplicated_columns = [x for x in ma_postconv_jorn.columns if "_xj" in x]
    # duplicated_columns.append('jrn_boberdoo_lead_id')
    # ma_postconv_jorn = ma_postconv_jorn.drop(columns=duplicated_columns)

    # ########

    # ### Zipcode Data ###

    # ## Get Zipcode Enriched Data
    # zip_data = get_zip_enrch_data(engine= hc_engine)

    # # zip_data = fill_and_drop_duplicates(
    # #     df= zip_data, 
    # #     identifiers=['zip_zipcode']
    # # )

    # zip_data = zip_data.groupby(by= "zip_zipcode").first().reset_index()

    # ## Merge Zipcode data with MA_POSTCONV_JORNAYA data
    # ma_postconv_jorn_zip = pd.merge(
    #     ma_postconv_jorn, 
    #     zip_data, 
    #     left_on= ['app_zip_code'],
    #     right_on= ['zip_zipcode'],
    #     how= 'left',
    #     suffixes= ("", "_xz")
    # )

    # ## Drop Duplicated Column names with "_z" suffix
    # duplicated_columns = [x for x in ma_postconv_jorn_zip.columns if "_xz" in x]
    # duplicated_columns.append('zip_zipcode')
    # ma_postconv_jorn_zip = ma_postconv_jorn_zip.drop(columns=duplicated_columns)

    # ## Some preprocessing on merged dataset
    # ma_postconv_jorn_zip['first_name'] = ma_postconv_jorn_zip["first_name"].str.lower()
    # ma_postconv_jorn_zip['last_name'] = ma_postconv_jorn_zip["last_name"].str.lower()

    # ma_phone_nums = list(ma_postconv_jorn_zip["owner_phone"].unique())
    # ma_first_names = list(ma_postconv_jorn_zip['first_name'].unique())
    # ma_last_names = list(ma_postconv_jorn_zip['last_name'].unique())

    # ########

    # ### Transunion Data ###

    # ## Get Transunion Data
    # tu_data = preprocess_transunion_data(
    #     username= "rutvik_bhende", 
    #     password= "0723@RutuJuly",
    #     account = "uza72979.us-east-1",
    #     phone_numbers_list= ma_phone_nums,
    #     first_names_list= ma_first_names,
    #     last_names_list= ma_last_names 
    # )

    # # tu_data_2 = fill_and_drop_duplicates(
    # #     df= tu_data,
    # #     identifiers= ['tu_PHONE_NUMBER']
    # # )

    # tu_data = tu_data.groupby(by= ['tu_PHONE_NUMBER', "tu_FIRST_NAME", "tu_LAST_NAME"]).first().reset_index()

    # ## Merge Trandunion data with MA_POSTCONV_JORNAYA_ZIP data
    # ma_postconv_jorn_zip_tu = pd.merge(
    #     ma_postconv_jorn_zip, 
    #     tu_data, 
    #     left_on= ['owner_phone', "first_name", "last_name"],
    #     right_on= ['tu_PHONE_NUMBER', "tu_FIRST_NAME", "tu_LAST_NAME"],
    #     how= 'left',
    #     suffixes= ("", "_xtu")
    # )

    # ## Drop Duplicated Column names with "_tu" suffix
    # duplicated_columns = [x for x in ma_postconv_jorn_zip_tu.columns if "_xtu" in x]
    # duplicated_columns.append('tu_PHONE_NUMBER')
    # duplicated_columns.append("tu_FIRST_NAME")
    # duplicated_columns.append("tu_LAST_NAME")
    # ma_postconv_jorn_zip_tu = ma_postconv_jorn_zip_tu.drop(columns=duplicated_columns)

    # ma_postconv_jorn_zip_tu = ma_postconv_jorn_zip_tu.drop_duplicates(keep='last', ignore_index=True)

    # ########


    # ## Modify LTV values. Replace Blanks with Zero.
    # ma_postconv_jorn_zip_tu['mod_LTV'] = (ma_postconv_jorn_zip_tu['post_raw_LTV'] / 1.95).fillna(0)

    # ## Save to file
    # ma_postconv_jorn_zip_tu.to_csv("data/ma_postconv_jorn_zip_tu.csv", index= False)


    ## Preprocess MA pre conversion data

    pp = preprocess_ma_data(
        data_path= "data/ma_postconv_jorn_zip_tu.csv"
    )

    print(pp.shape)



# def fill_and_drop_duplicates(df: pd.DataFrame, identifiers: Union[str, list]):
#     columns = list(df.columns)
#     [columns.remove(i) for i in list(identifiers)]

#     groups = df.groupby(identifiers)[columns].apply(
#         lambda x: x.ffill().bfill()
#     )
#     df.loc[:,columns] = groups.loc[:,columns]
#     df.drop_duplicates(keep="first", ignore_index=True, inplace=True)

#     return df





if __name__ == "__main__":
    main()
