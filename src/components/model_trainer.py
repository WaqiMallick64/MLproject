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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.loggers import logging
from xgboost import XGBRegressor

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_conf = ModelTrainerConfig()

    def initiateModelTrainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            x_train,y_train,x_test,y_test = (
                 train_array[:,:-1],
                 train_array[:,-1],
                 test_array[:,:-1],
                 test_array[:,-1]
             )

            models = {
                "RandomForest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbours Classifier" : KNeighborsRegressor(),
                "XGB classifier" : XGBRegressor(),
                "CatBoost Classifier": CatBoostRegressor(logging_level='Silent'),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models) 

            best_model_score = max(sorted(model_report.values()))

            for model_name , score in model_report.items():
                if  score == best_model_score:
                    best_model_name = model_name
                    break

            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found")
            logging.info(f"Best model on training and testing datasets: ")

            save_object(
                file_path=self.model_trainer_conf.trained_model_file_path,
                obj=best_model
            )  

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square        
        except Exception as e:
            raise CustomException(e,sys)        
        

