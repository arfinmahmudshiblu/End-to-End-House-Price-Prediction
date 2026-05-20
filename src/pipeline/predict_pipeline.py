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

    def predict(self, features):

        try:

            # ======================================================
            # Paths
            # ======================================================

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            print("Before Loading")

            # ======================================================
            # Load Model and Preprocessor
            # ======================================================

            model = load_object(
                file_path=model_path
            )

            preprocessor = load_object(
                file_path=preprocessor_path
            )

            print("After Loading")

            # ======================================================
            # Transform Input Data
            # ======================================================

            data_scaled = preprocessor.transform(features)

            # ======================================================
            # Prediction
            # ======================================================

            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

    # ======================================================
    # SHAP Explanation
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

            # ======================================================
            # Load Objects
            # ======================================================

            model = load_object(model_path)

            preprocessor = load_object(preprocessor_path)

            # ======================================================
            # Transform Features
            # ======================================================

            data_scaled = preprocessor.transform(features)

            # ======================================================
            # SHAP Explainer
            # ======================================================

            explainer = shap.Explainer(model)

            shap_values = explainer(data_scaled)

            # ======================================================
            # Feature Importance
            # ======================================================

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

        total_sf: int,

        house_age: int,

        gr_liv_area: int,

        first_flr_sf: int,

        garage_area: int,

        total_bsmt_sf: int,

        year_remod_add: int,

        year_built: int,

        fireplaces: int,

        # ======================================================
        # Categorical Features
        # ======================================================

        ms_zoning: str,

        neighborhood: str,

        house_style: str,

        exterior_1st: str,

        exter_qual: str,

        foundation: str,

        heating_qc: str,

        kitchen_qual: str,

        garage_type: str,

        sale_condition: str

    ):

        # ======================================================
        # Numerical Features
        # ======================================================

        self.overall_qual = overall_qual

        self.total_sf = total_sf

        self.house_age = house_age

        self.gr_liv_area = gr_liv_area

        self.first_flr_sf = first_flr_sf

        self.garage_area = garage_area

        self.total_bsmt_sf = total_bsmt_sf

        self.year_remod_add = year_remod_add

        self.year_built = year_built

        self.fireplaces = fireplaces

        # ======================================================
        # Categorical Features
        # ======================================================

        self.ms_zoning = ms_zoning

        self.neighborhood = neighborhood

        self.house_style = house_style

        self.exterior_1st = exterior_1st

        self.exter_qual = exter_qual

        self.foundation = foundation

        self.heating_qc = heating_qc

        self.kitchen_qual = kitchen_qual

        self.garage_type = garage_type

        self.sale_condition = sale_condition

    # ======================================================
    # Convert Data into DataFrame
    # ======================================================

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                # ======================================================
                # Numerical Features
                # ======================================================

                "Overall Qual": [self.overall_qual],

                "TotalSF": [self.total_sf],

                "HouseAge": [self.house_age],

                "Gr Liv Area": [self.gr_liv_area],

                "1st Flr SF": [self.first_flr_sf],

                "Garage Area": [self.garage_area],

                "Total Bsmt SF": [self.total_bsmt_sf],

                "Year Remod/Add": [self.year_remod_add],

                "Year Built": [self.year_built],

                "Fireplaces": [self.fireplaces],

                # ======================================================
                # Categorical Features
                # ======================================================

                "MS Zoning": [self.ms_zoning],

                "Neighborhood": [self.neighborhood],

                "House Style": [self.house_style],

                "Exterior 1st": [self.exterior_1st],

                "Exter Qual": [self.exter_qual],

                "Foundation": [self.foundation],

                "Heating QC": [self.heating_qc],

                "Kitchen Qual": [self.kitchen_qual],

                "Garage Type": [self.garage_type],

                "Sale Condition": [self.sale_condition]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)