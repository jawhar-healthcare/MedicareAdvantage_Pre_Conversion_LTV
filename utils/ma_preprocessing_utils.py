import pandas as pd
from sqlalchemy import column, text
import numpy as np
from datetime import date, datetime
import pathlib
from typing import Union

from utils.utils import load_data, get_age, get_age_range

from snowflake.sqlalchemy import URL
import snowflake.connector

from warnings import filterwarnings

filterwarnings("ignore")


def get_post_conversion_data(data_path: Union[str, pathlib.Path]):

    ## Load MA Post-Conversion LTV Model Predictions data
    post_conv_data = load_data(data_path=data_path)

    # ## Keep only relevant features from MA Post Conversion LTV Data
    # relevant_features = [
    #     "application_id",
    #     "policy_id",
    #     "Predicted LTV",
    # ]
    # irrelevant_features = [
    #     feat for feat in post_conv_data.columns if feat not in relevant_features
    # ]

    # ## Keep only relevant columns
    # post_conv_data = post_conv_data.drop(columns=irrelevant_features)

    ltv_feat = [feat for feat in post_conv_data.columns if "ltv" in feat.lower()]

    ## Convert LTV value Strings to Floats if not
    if post_conv_data[ltv_feat].dtypes.item() != float:
        post_conv_data[ltv_feat] = (
            post_conv_data[ltv_feat]
            .str.replace("$", "")
            .str.replace(",", "")
            .astype(float)
        )

    # policy_nulls = post_conv_data["policy_id"].fillna("not-available")
    # post_conv_data["policy_id"] = policy_nulls

    ## Add a prefix "post_raw_" to all columns
    post_conv_data = post_conv_data.add_prefix("post_raw_")

    post_conv_data = post_conv_data.drop_duplicates(keep="last", ignore_index=True)

    return post_conv_data


def get_MedAdv_data(engine, save_csv=False):

    ma_data_sql = """
    SELECT
        fapp.application_id,
        fapp.owner_email,
        fapp.application_name,
        fapp.policy_id,
        fapp.owner_id,
        fapp.owner_phone,
        fapp.sk_submitted_date,
        fapp.owner_name,
        fapp.sk_date_of_birth,
        fapp.zip_code AS app_zip_code,
        fpol.zip_code AS pol_zip_code,
        fapp.parent_application_id,
        fapp.sk_referral_flag,
        dprod.bk_product_type,
        dcarr.carrier,
        dgen.bk_gender,
        dstate.bk_state,
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
        INNER JOIN isc_dm_sales.d_state dstate
                        ON fapp.sk_state = dstate.sk_state
        LEFT JOIN isc_dm_sales.d_status ds
                        ON ds.sk_status = fpol.sk_policy_status
        LEFT JOIN isc_dm_sales.f_calls fc
                        ON fc.application_id = fapp.parent_application_id
    WHERE bk_product IN (1048, 1019, 1020, 1052);
    """

    ma_data = pd.read_sql_query(text(ma_data_sql), engine)

    ma_data = ma_data.groupby(by="application_id").first().reset_index()

    # policy_nulls = ma_data["policy_id"].fillna("not-available")
    # ma_data["policy_id"] = policy_nulls

    ma_data["first_name"] = ma_data["owner_name"].apply(lambda x: x.split(" ")[0])
    ma_data["last_name"] = ma_data["owner_name"].apply(lambda x: x.split(" ")[-1])

    ma_data.drop(columns=["owner_name"], inplace=True)

    ## Application Submitted Date Preprocessing
    ma_data["sk_submitted_date"] = pd.to_datetime(
        ma_data["sk_submitted_date"], format="%Y%m%d", errors="coerce"
    )
    ma_data["interaction_weekday"] = ma_data["sk_submitted_date"].dt.weekday
    ma_data["interaction_day"] = ma_data["sk_submitted_date"].dt.day
    ma_data["interaction_month"] = ma_data["sk_submitted_date"].dt.month
    # ma_data["submitted_year"] = ma_data["sk_submitted_date"].dt.year

    ma_data["OEP"] = ma_data["sk_submitted_date"].apply(
        lambda x: get_enrollment_periods(date=x, period="OEP")
    )
    ma_data["MA_OEP"] = ma_data["sk_submitted_date"].apply(
        lambda x: get_enrollment_periods(date=x, period="MA_OEP")
    )

    ma_data["SEP"] = ma_data["sk_submitted_date"].apply(
        lambda x: get_enrollment_periods(date=x, period="SEP")
    )

    ma_data = ma_data.drop(columns="sk_submitted_date")

    ## Phone numer and area code
    ma_data["owner_phone"] = ma_data["owner_phone"].apply(
        lambda x: x[:10] if x is not None and len(x) > 9 else np.nan
    )
    ma_data["area_code"] = ma_data["owner_phone"].apply(
        lambda x: x[:3] if x is not np.nan else np.nan
    )

    ## DoB and Age
    ma_data["sk_date_of_birth"] = pd.to_datetime(
        ma_data["sk_date_of_birth"], format="%Y%m%d", errors="coerce"
    )

    ma_data["age"] = ma_data["sk_date_of_birth"].apply(lambda x: get_age(x))
    ma_data["age_range"] = ma_data["age"].apply(lambda x: get_age_range(x))
    ma_data = ma_data.drop(columns="sk_date_of_birth")

    ## Application Zip Code
    ma_data["app_zip_code"] = ma_data["app_zip_code"].where(
        ma_data["app_zip_code"].str.len() < 6, ma_data["app_zip_code"].str[:5]
    )
    ma_data["app_zip_code"] = ma_data["app_zip_code"].fillna("None")

    ## Policy Zip Code
    ma_data["pol_zip_code"] = ma_data["pol_zip_code"].where(
        ma_data["pol_zip_code"].str.len() < 6, ma_data["pol_zip_code"].str[:5]
    )
    ma_data["pol_zip_code"] = (
        ma_data["pol_zip_code"].replace("None", "N/A").fillna("None")
    )

    if save_csv:
        ma_data.to_csv("data/ma_data.csv", index=False)

    return ma_data


def get_enrollment_periods(date: datetime, period: str):

    if period.lower() == "oep":
        oep = 0
        start = datetime(date.year, 10, 15)
        end = datetime(date.year, 12, 7)
        if start <= date <= end:
            oep = 1
        return int(oep)

    if period.lower() == "ma_oep":
        ma_oep = 0
        start = datetime(date.year, 1, 1)
        end = datetime(date.year, 3, 31)
        if start <= date <= end:
            ma_oep = 1
        return int(ma_oep)

    if period.lower() == "sep":
        sep = 0
        start = datetime(date.year, 4, 1)
        end = datetime(date.year, 10, 14)
        if start <= date <= end:
            sep = 1
        return int(sep)


def get_jornaya_data(leads: list, engine, save_csv=False):

    jor_sql = f"""
    SELECT tje.response_audit_authentic,
           tje.response_audit_consumer_five_minutes,
           tje.response_audit_consumer_hour,
           tje.response_audit_consumer_twelve_hours,
           tje.response_audit_consumer_twelve_consumer_day,
           tje.response_audit_consumer_week,
           tje.response_audit_data_integrity,
           tje.response_audit_device_five_minutes,
           tje.response_audit_device_hour,
           tje.response_audit_device_twelve_hours,
           tje.response_audit_device_day,
           tje.response_audit_device_week,
           tje.response_audit_consumer_dupe_check,
           tje.response_audit_entity_value,
           tje.response_audit_ip_five_minutes,
           tje.response_audit_ip_hour,
           tje.response_audit_ip_twelve_hours,
           tje.response_audit_ip_day,
           tje.response_audit_ip_week,
           tje.response_audit_lead_age,
           tje.response_audit_age,
           tje.response_audit_lead_duration,
           tje.response_audit_duration,
           tje.response_audit_lead_dupe_check,
           tje.response_audit_lead_dupe,
           tje.response_audit_lead_five_minutes,
           tje.response_audit_lead_hour,
           tje.response_audit_lead_twelve_hours,
           tje.response_audit_lead_day,
           tje.response_audit_lead_week,
           bb.leadid AS lead_id
    FROM tracking.jornaya_event tje
    LEFT JOIN boberdoo.boberdoo bb
        ON bb.tcpa_universal_id = tje.tcpa_universal_id
    WHERE bb.product = 'MEDICARE'
    AND leadid IN {leads};
    """

    jornaya = pd.read_sql_query(text(jor_sql), engine)

    ## Add a prefix "jrn_" to all columns
    jornaya = jornaya.add_prefix("jrn_")

    jornaya = jornaya.groupby(by="jrn_lead_id").first().reset_index()

    if save_csv:
        jornaya.to_csv("data/jornaya.csv", index=False)

    return jornaya


def get_zip_data(username: str, password: str, account: str, save_csv=False):

    connect = snowflake.connector.connect(
        user=username, password=password, account=account
    )
    cursor = connect.cursor()

    query_str = """
    SELECT * FROM PROD_STAGE.LEAD_INFO_VALIDATION.ZCTA_MASTER;
    """
    cursor.execute(query_str)
    zip_data = cursor.fetch_pandas_all()

    zip_data.drop(columns=["DB_CREATION_DATE_TIME"], inplace=True)

    zip_data = zip_data.add_prefix("ZCTA_")

    zip_data = zip_data.rename(columns={c: c.lower() for c in zip_data.columns})

    if save_csv:
        zip_data.to_csv("data/zcta.csv", index=False)

    return zip_data


def get_county_city_data(username: str, password: str, account: str):

    connect = snowflake.connector.connect(
        user=username, password=password, account=account
    )
    cursor = connect.cursor()

    query_str = """
    SELECT ztzc.ZCTA,
        ztzc.PO_NAME AS CITY,
        ztzc.STATE,
        zcr.FIPS,
        cm.COUNTY
    FROM PROD_STAGE.LEAD_INFO_VALIDATION.ZIPTOZCTA_CROSSWALK ztzc
        LEFT JOIN PROD_STAGE.LEAD_INFO_VALIDATION.ZCTA_COUNTY_RELATIONSHIP zcr
            ON ztzc.ZCTA = zcr.ZCTA
        INNER JOIN PROD_STAGE.LEAD_INFO_VALIDATION.COUNTY_MASTER cm
            ON zcr.FIPS = cm.FIPS;
    """
    cursor.execute(query_str)
    county_city = cursor.fetch_pandas_all()

    county_city = county_city.rename(
        columns={c: c.lower() for c in county_city.columns}
    )

    county_city["county"] = county_city["county"].apply(
        lambda x: x.replace(" County", "")
    )

    return county_city


def load_transunion_data(phones: list, username: str, password: str, account: str):
    """
    Loads the TransUnion data from the Snowflake Data Warehouse
    using the user specified credentials.
    Args:

    Returns:

    """
    connect = snowflake.connector.connect(
        user=username, password=password, account=account
    )
    cursor = connect.cursor()

    query_string = f"""
    SELECT FIRST_NAME,
        LAST_NAME,
        PHONE_NUMBER,
        ZIP,
        CITY,
        STATE,
        SCORES[0][2]::FLOAT AS contact_score,
        SCORES[1][2]::FLOAT AS credit_score,
        DEMO_AGE_YEARS,
        DEMO_INCOME_DOLLARS,
        DEMO_CHILDREN_YES,
        DEMO_CHILDREN_NO,
        DEMO_AFFILIATION_CONSERVATIVE,
        DEMO_AFFILIATION_LIBERAL,
        DEMO_EDUCATION_YEARS,
        DEMO_HOMEOWNER_YES,
        DEMO_HOMEOWNER_NO,
        DEMO_HOMEVALUE_DOLLARS,
        DEMO_RESIDENT_YEARS,
        DEMO_OCCUPATION_FIRST
    FROM PROD_STAGE.TRACKING.INTERNAL_TRANSUNION_EVENT;
    """
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
    save_csv=False,
):
    """ """
    # Load TU Data
    tu_data = load_transunion_data(
        phones=phone_numbers_list, username=username, password=password, account=account
    )

    # Change Case of Name Strings in TU DataFrame
    tu_data["FIRST_NAME"] = tu_data["FIRST_NAME"].str.lower()
    tu_data["LAST_NAME"] = tu_data["LAST_NAME"].str.lower()

    # Truncate the TU data to only existing entries of Phone numbers and Names
    trunc_tu_data = tu_data[tu_data["PHONE_NUMBER"].isin(phone_numbers_list)]
    trunc_tu_data = trunc_tu_data[trunc_tu_data["FIRST_NAME"].isin(first_names_list)]
    trunc_tu_data = trunc_tu_data[trunc_tu_data["LAST_NAME"].isin(last_names_list)]

    trunc_tu_data.reset_index(inplace=True)
    trunc_tu_data.drop(columns=["index"], inplace=True)

    ## Add a prefix "tu_" to all columns
    trunc_tu_data = trunc_tu_data.add_prefix("tu_")

    trunc_tu_data["tu_ZIP"] = trunc_tu_data["tu_ZIP"].where(
        trunc_tu_data["tu_ZIP"].str.len() < 6, trunc_tu_data["tu_ZIP"].str[:5]
    )

    trunc_tu_data.rename(
        columns={c: c.lower() for c in trunc_tu_data.columns}, inplace=True
    )

    if save_csv:
        trunc_tu_data.to_csv("data/trunc_tu_data.csv", index=False)

    return trunc_tu_data


def get_united_features(df: pd.DataFrame, features_with: list):

    for feat in features_with:
        common_features = [col for col in df.columns if feat in col.lower()]

        if df[common_features[0]].dtype in [object, str]:
            temp1 = df[common_features[0]].map(
                lambda x: str(x).upper(), na_action="ignore"
            )
        else:
            temp1 = df[common_features[0]]

        for i in range(len(common_features) - 1):
            if df[common_features[i + 1]].dtype in [object, str]:
                temp2 = df[common_features[i + 1]].map(
                    lambda x: str(x).upper(), na_action="ignore"
                )
                temp1 = temp1.combine_first(temp2)
            else:
                temp2 = df[common_features[i + 1]]
                temp1 = temp1.combine_first(temp2)

        df = df.drop(columns=common_features)
        df[feat] = temp1

    return df
