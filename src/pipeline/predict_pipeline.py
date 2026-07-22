import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


# ======================================================
# Prediction Pipeline
# ======================================================

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor...")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            logging.info("Transforming input data...")

            data_scaled = preprocessor.transform(features)

            logging.info("Predicting house price...")

            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


# ======================================================
# Custom Data Class
# ======================================================

class CustomData:

    def __init__(
        self,
        overall_qual,
        overall_cond,
        total_bsmt_sf,
        first_flr_sf,
        gr_liv_area,
        garage_area,
        ms_zoning,
        neighborhood,
        house_style,
        heating_qc,
        central_air,
        kitchen_qual,
        garage_type,
        sale_condition
    ):

        self.overall_qual = overall_qual
        self.overall_cond = overall_cond
        self.total_bsmt_sf = total_bsmt_sf
        self.first_flr_sf = first_flr_sf
        self.gr_liv_area = gr_liv_area
        self.garage_area = garage_area

        self.ms_zoning = ms_zoning
        self.neighborhood = neighborhood
        self.house_style = house_style
        self.heating_qc = heating_qc
        self.central_air = central_air
        self.kitchen_qual = kitchen_qual
        self.garage_type = garage_type
        self.sale_condition = sale_condition

    def get_data_as_dataframe(self):

        custom_data_input_dict = {
            "Overall Qual": [self.overall_qual],
            "Overall Cond": [self.overall_cond],
            "Total Bsmt SF": [self.total_bsmt_sf],
            "1st Flr SF": [self.first_flr_sf],
            "Gr Liv Area": [self.gr_liv_area],
            "Garage Area": [self.garage_area],
            "MS Zoning": [self.ms_zoning],
            "Neighborhood": [self.neighborhood],
            "House Style": [self.house_style],
            "Heating QC": [self.heating_qc],
            "Central Air": [self.central_air],
            "Kitchen Qual": [self.kitchen_qual],
            "Garage Type": [self.garage_type],
            "Sale Condition": [self.sale_condition]
        }

        return pd.DataFrame(custom_data_input_dict)
    