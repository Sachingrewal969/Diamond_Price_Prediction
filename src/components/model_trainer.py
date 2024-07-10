import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('splitting dependent and independent variable from train ans test')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet': ElasticNet(), 
                'DecisionTree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'AdaBoost classifier':AdaBoostRegressor(),
                'Random Forest':RandomForestRegressor(),
                'K nearest Neighbour':KNeighborsRegressor()
            }
            params = {
                'LinearRegression': {},
                'Lasso': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                'Ridge': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                },
                'ElasticNet': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 1.0]
                },
                'DecisionTree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'RandomForest': {
                    'n_estimators': [100, 200, 300],
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'KNeighbors': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params)
            print(model_report)
            print('\n====================================================')
            logging.info(f'Model Report : {model_report}')

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'best model found, model name : {best_model_name}, R2 Score : {best_model_score}')
            print('\n==============================================================')
            logging.info(f'Best model found,model name :{best_model_name},R2 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info('exception occured at model training')
            raise CustomException(e,sys)
