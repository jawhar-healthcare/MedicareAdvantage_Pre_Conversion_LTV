import pandas as pd
from sqlalchemy import text
import numpy as np 
import datetime

def get_MA_data(engine):

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
    
    return ma_data
