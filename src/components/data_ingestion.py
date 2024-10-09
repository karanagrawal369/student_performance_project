import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Data ingestion configuration class
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path for train data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path for test data
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path for raw data

# Data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from a specific location (update the path if needed)
            df = pd.read_csv('Data\studentperf.csv')
            logging.info('Read the dataset as dataframe')

            # Create directory to store artifacts if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into training and testing sets
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
