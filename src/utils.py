import os
import sys
import pickle
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save Python object using pickle.
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def save_dill_object(file_path, obj):
    """
    Save Python object using dill.
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load pickle object.
    """

    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_dill_object(file_path):
    """
    Load dill object.
    """

    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple models using GridSearchCV.
    """

    try:

        report = {}

        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            para = param[model_name]

            gs = GridSearchCV(
                model,
                para,
                cv=3
            )

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "train_r2_score": train_model_score,
                "test_r2_score": test_model_score,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression metrics.
    """

    try:

        mae = mean_absolute_error(y_true, y_pred)

        mse = mean_squared_error(y_true, y_pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }

    except Exception as e:
        raise CustomException(e, sys)


def dataframe_summary(df):
    """
    Return dataframe summary.
    """

    try:

        summary = pd.DataFrame({
            "DataType": df.dtypes,
            "MissingValues": df.isnull().sum(),
            "UniqueValues": df.nunique()
        })

        return summary

    except Exception as e:
        raise CustomException(e, sys)


def remove_outliers_iqr(df, column):
    """
    Remove outliers using IQR method.
    """

    try:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[
            (df[column] >= lower_bound) &
            (df[column] <= upper_bound)
        ]

        return filtered_df

    except Exception as e:
        raise CustomException(e, sys)