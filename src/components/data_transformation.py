import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ======================================================
# Config
# ======================================================

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# ======================================================
# Transformation Class
# ======================================================

class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    # ==================================================
    # Preprocessor
    # ==================================================

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "Overall Qual",
                "Overall Cond",
                "TotalBsmtSF",
                "1stFlrSF",
                "GrLivArea",
                "GarageArea"
            ]

            categorical_columns = [
                "MS Zoning",
                "Neighborhood",
                "House Style",
                "Heating QC",
                "Central Air",
                "Kitchen Qual",
                "Garage Type",
                "Sale Condition"
            ]

            logging.info(f"Numerical Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            # ---------------------------
            # Numerical Pipeline
            # ---------------------------
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # ---------------------------
            # Categorical Pipeline
            # ---------------------------
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            # ---------------------------
            # Column Transformer
            # ---------------------------
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ==================================================
    # Transformation
    # ==================================================

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train/Test loaded successfully")

            target = "SalePrice"

            preprocessor = self.get_data_transformer_object()

            # ---------------------------
            # Split features
            # ---------------------------
            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            # test may or may not have target
            if target in test_df.columns:
                X_test = test_df.drop(columns=[target])
                y_test = test_df[target]
            else:
                X_test = test_df
                y_test = None

            # ---------------------------
            # Fit transform train
            # ---------------------------
            X_train_arr = preprocessor.fit_transform(X_train)

            # transform test
            X_test_arr = preprocessor.transform(X_test)

            logging.info("Transformation completed")

            # ---------------------------
            # Save preprocessor
            # ---------------------------
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Preprocessor saved successfully")

            # ---------------------------
            # Return clean outputs
            # ---------------------------
            if y_test is not None:
                return (
                    np.c_[X_train_arr, y_train],
                    np.c_[X_test_arr, y_test],
                    self.config.preprocessor_obj_file_path
                )

            return (
                np.c_[X_train_arr, y_train],
                X_test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)