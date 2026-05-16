# import os
# import sys
# from dataclasses import dataclass

# import shap
# import numpy as np
# import pandas as pd

# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV

# from sklearn.linear_model import (
#     LinearRegression,
#     Ridge,
#     Lasso
# )

# from sklearn.ensemble import (
#     RandomForestRegressor,
#     GradientBoostingRegressor,
#     VotingRegressor,
#     StackingRegressor
# )

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor

# from xgboost import XGBRegressor

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join(
#         "artifacts",
#         "best_model.pkl"
#     )


# class ModelTrainer:

#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def evaluate_models(
#         self,
#         X_train,
#         y_train,
#         X_test,
#         y_test,
#         models,
#         params
#     ):

#         try:

#             report = {}

#             for model_name, model in models.items():

#                 logging.info(f"Training {model_name}")

#                 param_grid = params[model_name]

#                 gs = GridSearchCV(
#                     model,
#                     param_grid,
#                     cv=3,
#                     scoring="r2",
#                     n_jobs=-1
#                 )

#                 gs.fit(X_train, y_train)

#                 best_model = gs.best_estimator_

#                 best_model.fit(X_train, y_train)

#                 y_pred = best_model.predict(X_test)

#                 score = r2_score(y_test, y_pred)

#                 report[model_name] = score

#                 models[model_name] = best_model

#                 logging.info(
#                     f"{model_name} R2 Score: {score}"
#                 )

#             return report, models

#         except Exception as e:
#             raise CustomException(e, sys)

#     def shap_explainability(
#         self,
#         model,
#         X_train,
#         X_test
#     ):

#         try:

#             logging.info("Generating SHAP values")

#             explainer = shap.Explainer(model, X_train)

#             shap_values = explainer(X_test)

#             shap.summary_plot(
#                 shap_values,
#                 X_test,
#                 show=True
#             )

#         except Exception as e:
#             raise CustomException(e, sys)

#     def initiate_model_trainer(
#         self,
#         train_array,
#         test_array
#     ):

#         try:

#             logging.info(
#                 "Splitting train and test arrays"
#             )

#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1]
#             )

#             models = {

#                 "Linear Regression":
#                     LinearRegression(),

#                 "Ridge Regression":
#                     Ridge(),

#                 "Lasso Regression":
#                     Lasso(),

#                 "Decision Tree":
#                     DecisionTreeRegressor(),

#                 "Random Forest":
#                     RandomForestRegressor(),

#                 "XGBoost":
#                     XGBRegressor(),

                
#                 "Support Vector Regressor":
#                     SVR()
#             }

#             params = {

#                 "Linear Regression": {},

#                 "Ridge Regression": {
#                     "alpha": [0.01, 0.1, 1.0, 10]
#                 },

#                 "Lasso Regression": {
#                     "alpha": [0.001, 0.01, 0.1, 1]
#                 },

#                 "Decision Tree": {
#                     "max_depth": [3, 5, 10, None],
#                     "min_samples_split": [2, 5, 10]
#                 },

#                 "Random Forest": {
#                     "n_estimators": [50, 100, 200],
#                     "max_depth": [5, 10, None]
#                 },

#                 "XGBoost": {
#                     "learning_rate": [0.01, 0.05, 0.1],
#                     "n_estimators": [100, 200],
#                     "max_depth": [3, 5, 7]
#                 },
                
#                 "Support Vector Regressor": {
#                     "C": [0.1, 1, 10],
#                     "kernel": ["linear", "rbf"]
#                 }
#             }

#             model_report, trained_models = self.evaluate_models(
#                 X_train=X_train,
#                 y_train=y_train,
#                 X_test=X_test,
#                 y_test=y_test,
#                 models=models,
#                 params=params
#             )

#             logging.info(model_report)

#             # ---------------------------------------------------
#             # Best Base Model
#             # ---------------------------------------------------

#             best_model_score = max(model_report.values())

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(
#                     best_model_score
#                 )
#             ]

#             best_model = trained_models[best_model_name]

#             logging.info(
#                 f"Best Model: {best_model_name}"
#             )

#             # ---------------------------------------------------
#             # Voting Regressor
#             # ---------------------------------------------------

#             voting_regressor = VotingRegressor(
#                 estimators=[
#                     (
#                         "rf",
#                         trained_models["Random Forest"]
#                     ),
#                     (
#                         "xgb",
#                         trained_models["XGBoost"]
#                     )
#                 ]
#             )

#             voting_regressor.fit(X_train, y_train)

#             voting_pred = voting_regressor.predict(X_test)

#             voting_score = r2_score(
#                 y_test,
#                 voting_pred
#             )

#             logging.info(
#                 f"Voting Regressor R2 Score: {voting_score}"
#             )

#             # ---------------------------------------------------
#             # Stacking Regressor
#             # ---------------------------------------------------

#             stacking_regressor = StackingRegressor(
#                 estimators=[
#                     (
#                         "rf",
#                         trained_models["Random Forest"]
#                     ),
#                     (
#                         "xgb",
#                         trained_models["XGBoost"]
#                     )
                    
#                 ],
#                 final_estimator=LinearRegression()
#             )

#             stacking_regressor.fit(X_train, y_train)

#             stacking_pred = stacking_regressor.predict(
#                 X_test
#             )

#             stacking_score = r2_score(
#                 y_test,
#                 stacking_pred
#             )

#             logging.info(
#                 f"Stacking Regressor R2 Score: {stacking_score}"
#             )

#             # ---------------------------------------------------
#             # Save Best Model
#             # ---------------------------------------------------

#             final_model = stacking_regressor

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=final_model
#             )

#             logging.info(
#                 "Best model saved successfully"
#             )

#             # ---------------------------------------------------
#             # SHAP Explainability
#             # ---------------------------------------------------

#             self.shap_explainability(
#                 model=trained_models["XGBoost"],
#                 X_train=X_train,
#                 X_test=X_test
#             )

#             # ---------------------------------------------------
#             # Final Prediction Score
#             # ---------------------------------------------------

#             final_predictions = final_model.predict(
#                 X_test
#             )

#             final_r2_score = r2_score(
#                 y_test,
#                 final_predictions
#             )

#             logging.info(
#                 f"Final R2 Score: {final_r2_score}"
#             )

#             return final_r2_score

#         except Exception as e:
#             raise CustomException(e, sys)