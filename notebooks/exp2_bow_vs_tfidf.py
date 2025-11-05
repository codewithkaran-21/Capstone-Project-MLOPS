import os 
import re
import string
import joblib
import pandas as pd
import numpy 
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "test_size" : 0.2,
    "data_path" : "notebooks/IMDB Dataset.csv",
    "mlflow_tracking_uri" : "https://dagshub.com/codewithkaran-21/Capstone-Project-MLOPS.mlflow",
    "dagshub_repo_owner" :  "codewithkaran-21",
    "dagshub_repo_name" :  "Capstone-Project-MLOPS",
    "experimnent_name" : "BOW vs tfIDF"
}

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG['dagshub_repo_owner'] , repo_name=CONFIG['dagshub_repo_name'],mlflow=True)
mlflow.set_experiment(CONFIG['experimnent_name'])

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive' , 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative' : 0 , 'positive' : 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error occured loading data {e}")
        raise

VECTORIZERS = {
    "BoW" : CountVectorizer(),
    "TF-IDF" : TfidfVectorizer()
}

ALGORITHMS = {
    "LogisticRegression"  :LogisticRegression(),
    "RandomForestClassifier" : RandomForestClassifier(),
    "GradientBoostingClassifier" : GradientBoostingClassifier(),
    "MultinomialNB" : MultinomialNB(),
    "xgboost" : XGBClassifier()
}


def train_and_evaluate(df):
    with mlflow.start_run(run_name="All-Experiment") as parent_run:
        for algo_name , ALGORITHM in ALGORITHMS.items():
            for vec , vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec}" , nested=True) as child_run:
                    X = vectorizer.fit_transform(df['review'])
                    y = df['sentiment']
                    X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=CONFIG['test_size'] , random_state=42)
                    mlflow.log_params({
                        "vectorizer" : vec,
                        "algorithm" : algo_name,
                        "testsize" : CONFIG['test_size']}
                    )

                    model = ALGORITHM
                    model.fit(X_train , y_train)

                    log_model_params(algo_name , model)

                    y_pred = model.predict(X_test)

                    metrics = {
                        "Accuracy"  : accuracy_score(y_test , y_pred),
                        "precision"  :precision_score(y_test , y_pred),
                        "reacll"  : recall_score(y_test , y_pred),
                        "f1-score" : f1_score(y_test , y_pred)

                    }

                    mlflow.log_metrics(metrics)

                    input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                    # mlflow.sklearn.log_model(model , "model")
                    joblib.dump(model , "model.pkl")
                    mlflow.log_artifact("model.pkl" , artifact_path="model")
                    print(f"Algorithm : {algo_name}, Vectorizer : {vec}")
                    print(f"Metrics : {metrics}")

def log_model_params(algo_name , model):
    params_to_log = {}
    if algo_name == "LogisticRegression":
        params_to_log["C"] = model.C
    elif algo_name == "RandomForestClassifier":
        params_to_log["n_estimators"] = model.n_estimators 
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == "GradientBoostingClassifier":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == "MultinomialNB":
        params_to_log["alpha"] = model.alpha
    elif algo_name == "xgboost":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    
    mlflow.log_params(params_to_log)

if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)

