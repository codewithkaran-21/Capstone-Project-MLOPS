import numpy as np 
import pandas as pd 
import yaml 
import os 
from sklearn.feature_extraction.text import CountVectorizer
from src.logger import logging
import pickle

def load_params(param_path : str) -> dict:
    try:
        with open(param_path , "r") as f:
            params  = yaml.safe_load(f)
        logging.debug("Parameters loaded from %s", param_path)
        return params
    
    except FileNotFoundError:
        logging.error("File not found %s",param_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML ERROR : %s",e)
        raise
    except Exception as e:
        logging.error("File not found %s",param_path)
        raise

def load_data(file_path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logging.info("Data loaded and NANs filled from : %s",file_path)
        return df 
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV File %s",e)
        raise
    except Exception as e:
        logging.error("Failed to load the data")
        raise

def apply_bow(train_data : pd.DataFrame , test_data : pd.DataFrame , max_features  :int) -> tuple:
    try:
        logging.info("Applying BOW")
        vectorizer = CountVectorizer(max_features=max_features)
        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())   
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        pickle.dump(vectorizer , open('models/vectorizer.pkl', 'wb'))
        logging.info("Bag Of Words Applied and data is transformed")

        return train_df , test_df
    except Exception as e:
        logging.error("Unexpected error occured while doing Bag of Words %s",e)
        raise  
        
def save_data(df : pd.DataFrame , file_path : str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path , index=False)
        logging.info("Data saved to path : %s",file_path)

    except Exception as e:
        logging.error(f"Error occured while saving the data to path :{file_path}")
        raise

def main():
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']
        # max_features = 20

        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        train_df , test_df = apply_bow(train_data , test_data , max_features)
        save_data(train_df , os.path.join("./data","processed","train_bow.csv"))
        save_data(test_df , os.path.join("./data","processed","test_bow.csv"))
    except Exception as e:
        logging.error(f"Failed to complete the feature engineering process ")
        print(f"Error : {e}")

if __name__ == "__main__":
    main()



