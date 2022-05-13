import pandas as pd
from sqlalchemy import text


def get_MA_data(engine):

    ma_appls_sql = """
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

    ma_appls = pd.read_sql_query(text(ma_appls_sql), engine)
    
    ## Drop Duplicate instances
    ma_appls = ma_appls.drop_duplicates(keep='last', ignore_index=True)
    ma_appls.reset_index(inplace=True)
    ma_appls.drop(columns=['index'], inplace= True)

    return ma_appls
