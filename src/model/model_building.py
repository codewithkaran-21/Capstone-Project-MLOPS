import numpy as np 
import pandas as pd 
import yaml 
import os 
from sklearn.linear_model import LogisticRegression
from src.logger import logging
import pickle


def load_data(file_path : str) -> pd.DataFrame:
    """The method is used to load the data from the CSV File"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"File loaded successfully from the path : {file_path}")    
        return df 
    except Exception as e:
        logging.error("Failed to load the data from the path : %s",file_path)
        raise

def train_model(X_train : np.ndarray , y_train : np.ndarray) -> LogisticRegression:
    try:
        clf = LogisticRegression(C=10 , solver="liblinear",penalty='l2')
        clf.fit(X_train , y_train)
        logging.info("Model Training completed...")
        return clf
    except Exception as e:
        logging.error("Error during model training %s",e)
        raise

def save_model(model , file_path : str) -> None:
    try:
        with open(file_path , "wb") as file:
            pickle.dump(model,file)
        logging.info("Model saved to path : %s",file_path)
    except Exception as e:
        logging.error("Failed to save the model : %s",e)
        raise
def main():
    try:
        train_data = load_data("./data/processed/train_bow.csv")
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train , y_train)

        save_model(clf , "models/model.pkl")
        logging.info("Model saved...")
    except Exception as e:
        logging.error("Error occured while training model : %s",e)
        raise
if __name__ == "__main__":
    main()


