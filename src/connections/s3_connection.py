import boto3
import pandas as pd 
import logging
from src.logger import logging
from io import StringIO

class s3_operations:
    def __init__(self , bucket_name , aws_access_key , aws_secret_key , region_name = "us-east-1"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id = aws_access_key,
            aws_secret_access_key = aws_secret_key,
            region_name = region_name  
        )

        logging.info("Data Ingestion from S3 Bucket Intialized....")

    def fetch_file_from_s3(self ,file_key : str) -> pd.DataFrame:
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logging.info(f"Fetchinh file from {file_key} from s3 bucket : {self.bucket_name}...")
            obj = self.s3_client.get_object(Bucket = self.bucket_name , Key = file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded data from {file_key} that has size {len(df)} from s3")
            return df 
        except Exception as e:
            logging.exception(f"Failed to fetch data '{file_key} from s3 : {e}")
            return None
        
# if __name__ == "__main__":

#     FILE_KEY = "IMDB Dataset.csv"

#     data_ingestion = s3_operations(BUCKET_NAME , AWS_ACCESS_KEY , AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)
    