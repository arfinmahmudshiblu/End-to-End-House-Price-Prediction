from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

# ======================================================
# Flask Application
# ======================================================

application = Flask(__name__)

app = application


# ======================================================
# Home Route
# ======================================================

@app.route('/')
def index():

    return render_template('index.html')


# ======================================================
# Prediction Route
# ======================================================

@app.route('/predictdata', methods=['GET', 'POST'])

def predict_datapoint():

    if request.method == 'GET':

        return render_template('home.html')

    else:

        # ======================================================
        # Collect User Input
        # ======================================================

        data = CustomData(

            # Numerical Features

            overall_qual=float(
                request.form.get('Overall Qual')
            ),

            total_sf=float(
                request.form.get('TotalSF')
            ),

            house_age=float(
                request.form.get('HouseAge')
            ),

            gr_liv_area=float(
                request.form.get('Gr Liv Area')
            ),

            first_flr_sf=float(
                request.form.get('1st Flr SF')
            ),

            garage_area=float(
                request.form.get('Garage Area')
            ),

            total_bsmt_sf=float(
                request.form.get('Total Bsmt SF')
            ),

            year_remod_add=float(
                request.form.get('Year Remod/Add')
            ),

            year_built=float(
                request.form.get('Year Built')
            ),

            fireplaces=float(
                request.form.get('Fireplaces')
            ),

            # Categorical Features

            ms_zoning=request.form.get(
                'MS Zoning'
            ),

            neighborhood=request.form.get(
                'Neighborhood'
            ),

            house_style=request.form.get(
                'House Style'
            ),

            exterior_1st=request.form.get(
                'Exterior 1st'
            ),

            exter_qual=request.form.get(
                'Exter Qual'
            ),

            foundation=request.form.get(
                'Foundation'
            ),

            heating_qc=request.form.get(
                'Heating QC'
            ),

            kitchen_qual=request.form.get(
                'Kitchen Qual'
            ),

            garage_type=request.form.get(
                'Garage Type'
            ),

            sale_condition=request.form.get(
                'Sale Condition'
            )
        )

        # ======================================================
        # Convert Data into DataFrame
        # ======================================================

        pred_df = data.get_data_as_data_frame()

        print(pred_df)

        print("Before Prediction")

        # ======================================================
        # Prediction Pipeline
        # ======================================================

        predict_pipeline = PredictPipeline()

        print("Mid Prediction")

        # House Price Prediction
        results = predict_pipeline.predict(pred_df)

        print("After Prediction")

        # ======================================================
        # SHAP Value Prediction
        # ======================================================

        shap_df = predict_pipeline.shap_prediction(
            pred_df
        )

        # Top 10 Important Features
        shap_table = shap_df.head(10).to_html(
            classes='table table-striped',
            index=False
        )

        # ======================================================
        # Render Result
        # ======================================================

        return render_template(

            'home.html',

            results=round(results[0], 2),

            shap_table=shap_table
        )


# ======================================================
# Main Function
# ======================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        debug=True
    )