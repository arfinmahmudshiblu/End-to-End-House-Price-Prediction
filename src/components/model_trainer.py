import os
import sys

import shap
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
    StackingRegressor
)

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )

    shap_explainer_path = os.path.join(
        "artifacts",
        "shap_explainer.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing input data"
            )

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # =========================
            # Base Models
            # =========================

            models = {

                "Linear Regression": LinearRegression(),

                "Ridge Regression": Ridge(),

                "Lasso Regression": Lasso(),

                "Decision Tree": DecisionTreeRegressor(),

                "Random Forest": RandomForestRegressor(),

                "Support Vector Machine": SVR(),

                "XGBoost": XGBRegressor(
                    objective="reg:squarederror",
                    verbosity=0
                ),

                "LightGBM": LGBMRegressor()
            }

            # =========================
            # Hyperparameters
            # =========================

            params = {

                "Linear Regression": {},

                "Ridge Regression": {
                    "alpha": [0.01, 0.1, 1, 10]
                },

                "Lasso Regression": {
                    "alpha": [0.001, 0.01, 0.1, 1]
                },

                "Decision Tree": {
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [10, 20]
                },

                "Support Vector Machine": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },

                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                },

                "LightGBM": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200]
                }
            }

            # =========================
            # Evaluate Models
            # =========================

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            logging.info(f"Model Report: {model_report}")

            # =========================
            # =========================
            # Best Model Selection
            # =========================

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["test_r2_score"]
            )

            best_model_score = model_report[
                best_model_name
            ]["test_r2_score"]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            logging.info(
                f"Best Model Test R2 Score: {best_model_score}"
            )

            # =========================
            # Train Best Model
            # =========================

            best_model.fit(X_train, y_train)

            # =========================
            # Voting Regressor
            # =========================

            voting_regressor = VotingRegressor(
                estimators=[
                    ("rf", RandomForestRegressor()),
                    ("xgb", XGBRegressor(
                        objective="reg:squarederror",
                        verbosity=0
                    )),
                    ("lgbm", LGBMRegressor())
                ]
            )

            voting_regressor.fit(X_train, y_train)

            # =========================
            # Stacking Regressor
            # =========================

            stacking_regressor = StackingRegressor(
                estimators=[
                    ("rf", RandomForestRegressor()),
                    ("xgb", XGBRegressor(
                        objective="reg:squarederror",
                        verbosity=0
                    )),
                    ("lgbm", LGBMRegressor())
                ],
                final_estimator=LinearRegression()
            )

            stacking_regressor.fit(X_train, y_train)

            # =========================
            # Evaluate Stacking Model
            # =========================

            predicted = stacking_regressor.predict(
                X_test
            )

            r2_square = r2_score(
                y_test,
                predicted
            )

            logging.info(
                f"Stacking Regressor R2 Score: {r2_square}"
            )

            # =========================
            # Save Final Model
            # =========================

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=stacking_regressor
            )

            logging.info(
                "Final model saved successfully"
            )

            # =========================
            # SHAP Explainer
            # =========================

            logging.info(
                "Creating SHAP explainer"
            )

            explainer = shap.Explainer(
                stacking_regressor.predict,
                X_train
            )

            save_object(
                file_path=self.model_trainer_config.shap_explainer_path,
                obj=explainer
            )

            logging.info(
                "SHAP explainer saved successfully"
            )

            return r2_square

        except Exception as e:

            raise CustomException(e, sys)