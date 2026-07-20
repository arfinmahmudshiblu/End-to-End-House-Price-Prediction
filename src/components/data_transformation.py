import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ==========================================================
# Configuration
# ==========================================================

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# ==========================================================
# Data Transformation
# ==========================================================

class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, X_train):

        try:

            # Automatically detect feature types
            numerical_columns = X_train.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            categorical_columns = X_train.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            logging.info(f"Numerical Columns ({len(numerical_columns)}): {numerical_columns}")
            logging.info(f"Categorical Columns ({len(categorical_columns)}): {categorical_columns}")

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehotencoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ======================================================
    # Data Transformation
    # ======================================================

    def initiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test datasets loaded successfully")

            target_column = "SalePrice"

            if target_column not in train_df.columns:
                raise ValueError(f"{target_column} column not found.")

            # Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            if target_column in test_df.columns:
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]
            else:
                X_test = test_df
                y_test = None

            # Create preprocessor
            preprocessing_obj = self.get_data_transformer_object(X_train)

            logging.info("Applying preprocessing...")

            X_train_processed = preprocessing_obj.fit_transform(X_train)
            X_test_processed = preprocessing_obj.transform(X_test)

            logging.info("Preprocessing completed successfully")

            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor saved successfully")

            train_arr = np.c_[X_train_processed, np.array(y_train)]

            if y_test is not None:
                test_arr = np.c_[X_test_processed, np.array(y_test)]
            else:
                test_arr = X_test_processed

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)