import sys
import os

import pandas as pd
import shap
import numpy as np

from src.exception import CustomException
from src.utils import load_object


# ======================================================
# Prediction Pipeline
# ======================================================

class PredictPipeline:

    def __init__(self):
        pass

    # ======================================================
    # House Price Prediction
    # ======================================================

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            print("Before Loading")

            # Load Model & Preprocessor
            model = load_object(
                file_path=model_path
            )

            preprocessor = load_object(
                file_path=preprocessor_path
            )

            print("After Loading")

            # Transform Input Data
            data_scaled = preprocessor.transform(
                features
            )

            # Prediction
            preds = model.predict(
                data_scaled
            )

            return preds

        except Exception as e:
            raise CustomException(e, sys)

    # ======================================================
    # SHAP Feature Importance
    # ======================================================

    def shap_prediction(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            # Load Objects
            model = load_object(model_path)

            preprocessor = load_object(preprocessor_path)

            # Transform Features
            data_scaled = preprocessor.transform(
                features
            )

            # SHAP Explainer
            explainer = shap.Explainer(model)

            shap_values = explainer(
                data_scaled
            )

            # Feature Importance DataFrame
            feature_importance = pd.DataFrame({

                "Feature":
                preprocessor.get_feature_names_out(),

                "SHAP Value":
                shap_values.values[0]

            })

            feature_importance["ABS_SHAP"] = np.abs(
                feature_importance["SHAP Value"]
            )

            feature_importance = feature_importance.sort_values(

                by="ABS_SHAP",

                ascending=False
            )

            return feature_importance

        except Exception as e:
            raise CustomException(e, sys)


# ======================================================
# Custom Data Class
# ======================================================

class CustomData:

    def __init__(

        self,

        # ======================================================
        # Numerical Features
        # ======================================================

        overall_qual: int,

        overall_cond: int,

        total_bsmt_sf: float,

        first_flr_sf: int,

        gr_liv_area: int,

        garage_area: float,

        # ======================================================
        # Categorical Features
        # ======================================================

        ms_zoning: str,

        neighborhood: str,

        house_style: str,

        heating_qc: str,

        central_air: str,

        kitchen_qual: str,

        garage_type: str,

        sale_condition: str

    ):

        # ======================================================
        # Numerical Features
        # ======================================================

        self.overall_qual = overall_qual

        self.overall_cond = overall_cond

        self.total_bsmt_sf = total_bsmt_sf

        self.first_flr_sf = first_flr_sf

        self.gr_liv_area = gr_liv_area

        self.garage_area = garage_area

        # ======================================================
        # Categorical Features
        # ======================================================

        self.ms_zoning = ms_zoning

        self.neighborhood = neighborhood

        self.house_style = house_style

        self.heating_qc = heating_qc

        self.central_air = central_air

        self.kitchen_qual = kitchen_qual

        self.garage_type = garage_type

        self.sale_condition = sale_condition

    # ======================================================
    # Convert Input into DataFrame
    # ======================================================

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                # ======================================================
                # Numerical Features
                # ======================================================

                "Overall Qual": [self.overall_qual],

                "Overall Cond": [self.overall_cond],

                "Total Bsmt SF": [self.total_bsmt_sf],

                "1st Flr SF": [self.first_flr_sf],

                "Gr Liv Area": [self.gr_liv_area],

                "Garage Area": [self.garage_area],

                # ======================================================
                # Categorical Features
                # ======================================================

                "MS Zoning": [self.ms_zoning],

                "Neighborhood": [self.neighborhood],

                "House Style": [self.house_style],

                "Heating QC": [self.heating_qc],

                "Central Air": [self.central_air],

                "Kitchen Qual": [self.kitchen_qual],

                "Garage Type": [self.garage_type],

                "Sale Condition": [self.sale_condition]
            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:
            raise CustomException(e, sys)