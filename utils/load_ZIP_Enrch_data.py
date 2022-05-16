import pandas as pd
from sqlalchemy import text

def get_zip_enrch_data(
    engine
):

    zip_enriched_sql = """
    SELECT * FROM data_science.zipcode_data_enriched;
    """
    zip_ = pd.read_sql_query(text(zip_enriched_sql), engine)

    ## Drop all Duplicated row entries
    zip_ = zip_.drop_duplicates(keep='last', ignore_index=True)
    zip_.reset_index(inplace=True)
    zip_.drop(columns=['index'], inplace= True)

    ## Add a prefix "zip_" to all columns
    zip_ = zip_.add_prefix("zip_")
    
    return zip_
