import numpy as np 
import pandas as pd 
import json 
import os 
import logging
from sklearn.metrics import accuracy_score , precision_score , recall_score ,f1_score , roc_auc_score
import pickle
from src.logger import logging
import mlflow
import dagshub
import mlflow.sklearn

# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "codewithkaran-21"
# repo_name = "Capstone-Project-MLOPS"

#  Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

mlflow.set_tracking_uri("https://dagshub.com/codewithkaran-21/Capstone-Project-MLOPS.mlflow")
dagshub.init(repo_owner='codewithkaran-21', repo_name='Capstone-Project-MLOPS', mlflow=True)

def load_model(file_path : str):
    try:
        with open(file_path , "rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully %s",file_path)
        return model
    except Exception as e:
        logging.error("Unexpected error occured while loading the model from path: %s",e)
        raise
    except FileNotFoundError:
        logging.error("File not Found: %s",file_path)
        raise
def load_data(file_path  : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from the path %s",file_path)
        return df 
    except Exception as e:
        logging.error("File not found from the path : %s",file_path)
        raise
def evaluate_model(clf , X_test : np.ndarray , y_test : np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test , y_pred)
        logging.info("Accuracy %s:",accuracy)
        precision = precision_score(y_test , y_pred)
        logging.info(f"Precision score : {precision}")
        recall = recall_score(y_test , y_pred)
        logging.info(f"Reacll Score : {recall}")
        f1 = f1_score(y_test , y_pred)
        logging.info(f"F1-Score : {f1}")
        auc = roc_auc_score(y_test , y_pred_proba)
        logging.info(f"roc_auc_score : {auc}")

        metrics_dict = {
            "accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall,
            "f1_score" : f1,
            "roc_auc_score" : auc
        }

        logging.info("Model Evaluation metrics evaluated")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error occured while processing the evaluation metrics")
        raise
def save_metric(metrics: dict , file_path : str) -> None:
    try:
        with open(file_path , "w") as file:
            json.dump(metrics , file , indent=4)
        logging.info("Metric saved at path : %s",file_path)

    except Exception as e:
        logging.error(f"Error occured while saving the metrics :",e)
        raise
def save_model_info(run_id : str , model_path : str , file_path : str) -> None:
    try:
        model_info = {"run_id" : run_id , "model_path" : model_path}
        with open(file_path , "w") as file:
            json.dump(model_info , file , indent=4)
        logging.info("Model info saved at %s",file_path)
    except Exception as e:
        logging.error("Unexpected occued while saving the model deatils")
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            metrics = evaluate_model(clf , X_test , y_test)
            save_metric(metrics , "reports/metrics.json")

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.sklearn.log_model(clf, "model")
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

