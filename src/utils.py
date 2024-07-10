import os,sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for model_name,model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            param=param.get(model_name,{})
            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            # make prediction
            y_test_pred=model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)
            
            report[model_name] = test_model_score
        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)

