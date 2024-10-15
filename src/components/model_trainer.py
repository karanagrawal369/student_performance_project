import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Log shapes of the datasets
            logging.info(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

            # Define the models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Simplified parameter grid for faster evaluation
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16],
                    'subsample': [0.8, 0.9],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                    'iterations': [30, 50],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16],
                }
            }

            logging.info("Initiating model evaluation...")
            # Evaluate all models
            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )

            # Log the model evaluation report
            logging.info(f"Model report: {model_report}")

            # Get best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with a satisfactory score")

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

            # Predict on the test set
            predicted = best_model.predict(X_test)

            # Calculate the R2 score
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score: {r2_square}")

            return r2_square

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            raise CustomException(e, sys)
