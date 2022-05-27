import pandas as pd
from sqlalchemy import text
import numpy as np 
import datetime

from snowflake.sqlalchemy import URL
import snowflake.connector

from warnings import filterwarnings
filterwarnings("ignore")


def preprocess_post_conversion_ltv_data(data_path: str):

    ## Load MA Post-Conversion LTV Model Predictions data
    post_conv_data = pd.read_csv(data_path, low_memory=False)

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

    ## Convert LTV value Strings to Floats
    post_conv_data["LTV"] = post_conv_data["LTV"].str.replace('$', '').str.replace(',', '').astype(float)

    ## Add a prefix "post_raw_" to all columns
    post_conv_data = post_conv_data.add_prefix("post_raw_")

    post_conv_data = post_conv_data.drop_duplicates(keep='last', ignore_index=True)

    return post_conv_data


def get_MA_data(engine, save_csv= False):

    ma_data_sql = """
    SELECT
        fapp.application_id,
        fapp.owner_email,
        fapp.application_name,
        fapp.policy_id,
        fapp.owner_id,
        fapp.owner_phone,
        fapp.sk_submitted_date,
        fapp.sk_owner_gender,
        fapp.owner_name,
        fapp.sk_date_of_birth,
        fapp.zip_code AS app_zip_code,
        fpol.zip_code AS pol_zip_code,
        fapp.parent_application_id,
        fapp.sk_referral_flag,
        dprod.bk_product_type,
        dcarr.carrier,
        dgen.bk_gender,
        fc.lead_id
    FROM isc_dm_sales.f_application fapp
        LEFT JOIN isc_dm_sales.f_policy fpol
                        ON fapp.policy_id=fpol.policy_id
        INNER JOIN isc_dm_sales.d_product dprod
                        ON fapp.sk_product = dprod.sk_product
        INNER JOIN isc_dm_sales.d_carrier dcarr
                        ON fapp.sk_carrier = dcarr.sk_carrier
        INNER JOIN isc_dm_sales.d_gender dgen
                        ON fapp.sk_owner_gender = dgen.sk_gender
        LEFT JOIN isc_dm_sales.d_status ds
                        ON ds.sk_status = fpol.sk_policy_status
        LEFT JOIN isc_dm_sales.f_calls fc
                        ON fc.application_id = fapp.parent_application_id
    WHERE bk_product IN (1048, 1019, 1020, 1052);
    """

    ma_data = pd.read_sql_query(text(ma_data_sql), engine)

    ma_data = ma_data.groupby(by= "application_id").first().reset_index()

    ma_data['first_name'] = ma_data["owner_name"].apply(lambda x: x.split(" ")[0])
    ma_data['last_name'] = ma_data["owner_name"].apply(lambda x: x.split(" ")[-1])

    ma_data.drop(columns=['owner_name'], inplace= True)

    ma_data['sk_submitted_date'] = pd.to_datetime(
        ma_data['sk_submitted_date'], 
        format="%Y%m%d",
        errors= "coerce"
    )
    ma_data["submitted_month"] = ma_data['sk_submitted_date'].dt.month
    ma_data["submitted_year"] = ma_data['sk_submitted_date'].dt.year

    ma_data["owner_phone"] = ma_data["owner_phone"].apply(
        lambda x: x[:10] if x is not None and len(x) > 9 else np.nan
    )

    ma_data["area_code"] = ma_data["owner_phone"].apply(
        lambda x: x[:3] if x is not np.nan else np.nan
    )

    ma_data['sk_date_of_birth'] = pd.to_datetime(
        ma_data['sk_date_of_birth'], 
        format="%Y%m%d",
        errors= "coerce"
    )

    ma_data['age'] = ma_data['sk_date_of_birth'].apply(
        lambda x: int(
            (
                datetime.datetime.now().date() - datetime.datetime.date(x)
            ).days / 365.2425
        )
    )

    ma_data['age_range'] = ma_data['age'].apply(
        lambda x: get_age_range(x)
    )

    ma_data["app_zip_code"] = ma_data["app_zip_code"].where(
        ma_data['app_zip_code'].str.len() < 6, 
        ma_data['app_zip_code'].str[:5]
    )

    ma_data["pol_zip_code"] = ma_data["pol_zip_code"].where(
        ma_data['pol_zip_code'].str.len() < 6, 
        ma_data['pol_zip_code'].str[:5]
    )

    if save_csv:
        ma_data.to_csv("data/ma_data.csv", index=False)

    
    return ma_data

def get_age_range(age: int or float):
    if age < 65:
        age_s = "Less than 65"
    elif age >= 65 and age < 75:
        age_s = "65 to 75"
    elif age >= 75 and age < 85:
        age_s = "75 to 85"
    elif age >= 85:
        age_s = "More than 85"
    else:
        age_s = "Undefined"
    return age_s


def get_jornaya_data(
    leads: list,
    engine,
    save_csv= False
):

    jor_sql = f"""
    WITH jornaya_trunc AS (
    SELECT *
    FROM tracking.jornaya_event
    ),
    boberdoo_publisher AS (
        SELECT leadid AS lead_id,
            product,
            lead_created,
            age,
            amount,
            state,
            source,
            lead_type,
            tcpa_universal_id
        FROM boberdoo.boberdoo
        WHERE product = 'MEDICARE'
        AND leadid IN {leads}
    ) SELECT bp.lead_id AS boberdoo_lead_id,
            bp.state,
            bp.amount AS boberdoo_amount,
            bp.source AS boberdoo_source,
            bp.lead_type AS boberdoo_lead_type,
            tje.*
    FROM boberdoo_publisher bp
    LEFT JOIN jornaya_trunc tje
        ON bp.tcpa_universal_id = tje.tcpa_universal_id;
    """

    jornaya = pd.read_sql_query(text(jor_sql), engine)

    ## Add a prefix "jrn_" to all columns
    jornaya = jornaya.add_prefix("jrn_")

    jornaya = jornaya.groupby(by= "jrn_boberdoo_lead_id").first().reset_index()

    ## Drop irrelevant features
    irrelevant_features = [
        "jrn_tracking_file_path",
        "jrn_request_age",
        "jrn_request_tcpa_universal_id",
        "jrn_request_provider",
        "jrn_request_dob",
        "jrn_id",
        "jrn_tcpa_universal_id",
        "jrn_url",
        "jrn_request_f_name",
        "jrn_request_l_name",
        "jrn_request_email",
        "jrn_request_phone1",
        "jrn_request_address1",
        "jrn_response_audit_token"
    ]

    ## Remove any zip feats 
    irrelevant_features = irrelevant_features \
                            + [z for z in jornaya.columns if "zip" in z.lower()] \
                                + [z for z in jornaya.columns if z.endswith("rule")]

    jornaya = jornaya.drop(columns= irrelevant_features)

    if save_csv:
        jornaya.to_csv("data/jornaya.csv", index=False)

    return jornaya


def get_zip_enrch_data(
    engine, 
    save_csv= False
):

    zip_sql = """
    SELECT * FROM data_science.zcta_data;
    """
    zip_ = pd.read_sql_query(text(zip_sql), engine)

    ## Add a prefix "zip_" to all columns
    zip_ = zip_.add_prefix("zcta_")

    if save_csv:
        zip_.to_csv("data/zcta.csv", index=False)

    return zip_


def load_transunion_data(
    username: str,
    password: str,
    account: str
):
    """
    Loads the TransUnion data from the Snowflake Data Warehouse
    using the user specified credentials.
    Args:

    Returns:

    """
    connect = snowflake.connector.connect(
        user= username, 
        password= password,
        account= account
    )
    cursor = connect.cursor()
    
    query_string = '''
        select *,
        SCORES[0][2]::FLOAT AS contact_score,
        SCORES[1][2]::FLOAT AS credit_score 
        from PROD_STAGE.WEB_TRACKING.INTERNAL_TRANSUNION_EVENT
    '''
    cursor.execute(query_string)

    tu_data = cursor.fetch_pandas_all()

    return tu_data


def preprocess_transunion_data(
    username: str,
    password: str,
    account: str,
    phone_numbers_list: list,
    first_names_list: list,
    last_names_list: list, 
    save_csv= False
):
    """
    """
    # Load TU Data
    tu_data = load_transunion_data(
        username= username,
        password= password,
        account= account
        )

    # Change Case of Name Strings in TU DataFrame
    tu_data['FIRST_NAME'] = tu_data['FIRST_NAME'].str.lower()
    tu_data['LAST_NAME'] = tu_data['LAST_NAME'].str.lower()

    # Truncate the TU data to only existing entries of Phone numbers and Names
    trunc_tu_data = tu_data[tu_data['PHONE_NUMBER'].isin(phone_numbers_list)]
    trunc_tu_data = trunc_tu_data[trunc_tu_data['FIRST_NAME'].isin(first_names_list)]
    trunc_tu_data = trunc_tu_data[trunc_tu_data['LAST_NAME'].isin(last_names_list)]
    
    trunc_tu_data.reset_index(inplace=True)
    trunc_tu_data.drop(columns=['index'], inplace= True)

    ## Add a prefix "tu_" to all columns
    trunc_tu_data = trunc_tu_data.add_prefix("tu_")

    ## Drop irrelevant features
    irrelevant_features = [
        "tu_REQUEST",
        "tu_EMAIL",
        "tu_DATA_INPUT",
        "tu_DOB",
        "tu_DEMO_AGE_YEARS",
        "tu_ADDRESS",
        "tu_STATE",
        "tu_STATUS_ID",
        "tu_STATUS_RESULT",
        "tu_MESSAGES",
        "tu_SCORES",
        "tu_API",
        "tu_SESSION_ID",
        "tu_TRACKING_DATE",
    ]

    ## Remove any irrelevant features 
    irrelevant_features = irrelevant_features \
        + [t for t in trunc_tu_data.columns if "verify" in t.lower()] \
            + [t for t in trunc_tu_data.columns if "match" in t.lower()]

    trunc_tu_data = trunc_tu_data.drop(columns= irrelevant_features)

    if save_csv:
        trunc_tu_data.to_csv("data/trunc_tu_data.csv", index=False)

    return trunc_tu_data