import pandas as pd
from sqlalchemy import text

def get_jornaya_data(
    leads: list,
    engine
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
            bp.product AS boberdoo_product,
            bp.age AS boberdoo_age,
            bp.state,
            bp.amount AS boberdoo_amount,
            bp.source AS boberdoo_source,
            bp.lead_type AS boberdoo_lead_type,
            bp.tcpa_universal_id AS boberdoo_tcpa_universal_id,
            tje.*
    FROM boberdoo_publisher bp
    LEFT JOIN jornaya_trunc tje
        ON bp.tcpa_universal_id = tje.tcpa_universal_id;
    """

    jornaya = pd.read_sql_query(text(jor_sql), engine)

    ## Drop all Duplicated row entries
    jornaya = jornaya.drop_duplicates(keep='last', ignore_index=True)
    jornaya.reset_index(inplace=True)
    jornaya.drop(columns=['index'], inplace= True)

    ## Add a prefix "jrn_" to all columns
    jornaya = jornaya.add_prefix("jrn_")

    return jornaya
