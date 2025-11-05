import os 
import string
import re
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import dagshub
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score ,  precision_score , recall_score , f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import  warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

CONFIG = {
    "test_size" : 0.2,
    "data_path" : "notebooks/IMDB Dataset.csv",
    "mlflow_tracking_uri" : "https://dagshub.com/codewithkaran-21/Capstone-Project-MLOPS.mlflow",
    "dagshub_repo_owner" :  "codewithkaran-21",
    "dagshub_repo_name" :  "Capstone-Project-MLOPS",
    "experimnent_name" : "LoR Hyparmeter tuning"
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG['dagshub_repo_owner'] , repo_name=CONFIG['dagshub_repo_name'],mlflow=True)
mlflow.set_experiment(CONFIG['experimnent_name'])

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])

    return text.strip()

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    df['review'] = df['review'].astype(str).apply(preprocess_text)
    df = df[df['sentiment'].isin(["positive" , "negative"])]

    df['sentiment'] = df['sentiment'].map({"positive" : 1 , "negative" : 0})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    return train_test_split(X , y , test_size=0.2 , random_state=42) , vectorizer

def train_and_log_model(X_train , X_test , y_train , y_test , vectorizer):
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression() , param_grid , cv=5 , scoring="f1" , n_jobs=-1)
        grid_search.fit(X_train , y_train)

        for params , mean_Score , std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"]
        ):
            with mlflow.start_run(run_name=f"LR with params {params}" , nested= True):
                model = LogisticRegression(**params)
                model.fit(X_train , y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy" : accuracy_score(y_test , y_pred),
                    "precision" : precision_score(y_test , y_pred),
                    "recall" : recall_score(y_test , y_pred),
                    "f1_score" : f1_score(y_test , y_pred),
                    "mean_cv_score" : mean_Score,
                    "std_cv_score" : std_score
                }
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_
        mlflow.log_params(best_params)
        mlflow.log_metric("f1-score" , best_f1)

        joblib.dump(best_model , "model.pkl")
        mlflow.log_artifact("model.pkl" , artifact_path="model")

        print(f"Best Params : {best_params} and f1-score {best_f1}")
        print(metrics)

if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(CONFIG["data_path"])
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)