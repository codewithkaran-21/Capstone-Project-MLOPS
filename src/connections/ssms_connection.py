import pyodbc
import pandas as pd 
import json
import os 


def main(config_path : str = "config.json") -> pd.DataFrame:
    """
    Fetches data from a SQL Server table and returns it as a DataFrame.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script dir : {script_dir}")
    config_file  = os.path.join(script_dir , config_path)
    print(f"Config path : {config_file}")

    with open(config_file , "r") as file:
        config = json.load(file)

    server = config['sql_server']['server']
    database = config['sql_server']['database']
    table = config['sql_server']['table']

    print(f"Server : {server} database : {database} table : {table}")

    connection_String = (
        f"DRIVER = {{SQL Server}};"
        f"SERVER = {server};"
        f"DATABASE = {database};"
        f"Trusted_Connection = yes;"
    )

    print(f"{connection_String}")

    try:
        conn = pyodbc.connect(connection_String)
        if conn:
            print("Connection to  SQL Sever Successfull")
        else:
            print("Could not connect to ssms")

        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query , conn)
        conn.close()
        print(f"Data Fetched from table { table}")
        return df 
    except Exception as e:
        print("Error occured while loading data from SQL Server" , e)
        raise None
