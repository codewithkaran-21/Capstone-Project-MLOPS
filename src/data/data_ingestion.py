import numpy
import os 
import pandas as pd 
pd.set_option('future.no_silent_downcasting', True)
import yaml 
import logging
from src.logger import logging
from sklearn.model_selection import train_test_split

def load_params(params_path : str) -> dict:
    try:
        with open(params_path , 'r') as file:
            params = yaml.safe_load(file)

        logging.debug(f"Params loaded from thr path %s , {params_path}")

    except FileNotFoundError:
        logging.error("File not found %s " , params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML error : %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error : %s",e)
        raise

def load_data(data_url : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loaded from %s",data_url)
        return df
    except pd.errors.ParserError as e:
        logging.info("Failed to parse CSV File : %s",e)
        raise
    except Exception as e:
        logging.info("Unexpected error occured while loading data :%s",e)
        raise

def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("Pre-processing Started.....")
        # df.drop(columns=["tweet_id"] , inplace=True)
        final_df = df[df['sentiment'].isin(["positive" , "negative"])]
        final_df['sentiment'] = final_df['sentiment'].replace({"positive" : 1 , "negative" : 0})
        logging.info("PreProcessing Completed.....")
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe %s",e)
        raise
    except Exception as e:
        logging.error("unexpected error occured during preproicessing %s",e)
        raise

def save_data(train_data : pd.DataFrame , test_data : pd.DataFrame , data_path : str) -> None:
    try:
        raw_data_path = os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logging.debug("Train and test data saved to %s " , raw_data_path)
    except Exception