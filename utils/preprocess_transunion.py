import numpy as np 
import pandas as pd
from snowflake.sqlalchemy import URL
import snowflake.connector

from warnings import filterwarnings
filterwarnings("ignore")

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
    last_names_list: list
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

    return trunc_tu_data


# if __name__ == "__main__":
#     tu = load_transunion_data(
#         query_string= '''
#             select *,
#             SCORES[0][2]::FLOAT AS contact_score,
#             SCORES[1][2]::FLOAT AS credit_score 
#         from PROD_STAGE.WEB_TRACKING.INTERNAL_TRANSUNION_EVENT
#         ''',
#         username= "rutvik_bhende", 
#         password= "0723@RutuJuly",
#         account = "uza72979.us-east-1"
#     )

#     print(tu)






