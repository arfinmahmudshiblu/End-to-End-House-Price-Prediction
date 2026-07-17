from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

# ======================================================
# Flask App Initialization
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

        try:

            # ==================================================
            # Collect Input Data From Form
            # ==================================================

            data = CustomData(

                # ==============================================
                # Numerical Features
                # ==============================================

                overall_qual=int(
                    request.form.get('Overall Qual')
                ),

                overall_cond=int(
                    request.form.get('Overall Cond')
                ),

                total_bsmt_sf=float(
                    request.form.get('Total Bsmt SF')
                ),

                first_flr_sf=int(
                    request.form.get('1st Flr SF')
                ),

                gr_liv_area=int(
                    request.form.get('Gr Liv Area')
                ),

                garage_area=float(
                    request.form.get('Garage Area')
                ),

                # ==============================================
                # Categorical Features
                # ==============================================

                ms_zoning=request.form.get(
                    'MS Zoning'
                ),

                neighborhood=request.form.get(
                    'Neighborhood'
                ),

                house_style=request.form.get(
                    'House Style'
                ),

                heating_qc=request.form.get(
                    'Heating QC'
                ),

                central_air=request.form.get(
                    'Central Air'
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

            # ==================================================
            # Convert Input Into DataFrame
            # ==================================================

            pred_df = data.get_data_as_data_frame()

            print("\nInput DataFrame")
            print(pred_df)

            # ==================================================
            # Prediction Pipeline
            # ==================================================

            predict_pipeline = PredictPipeline()

            # ==================================================
            # House Price Prediction
            # ==================================================

            results = predict_pipeline.predict(
                pred_df
            )

            predicted_price = round(
                results[0],
                2
            )

            print("\nPredicted Price:")
            print(predicted_price)

            # ==================================================
            # SHAP Feature Importance
            # ==================================================

            shap_df = predict_pipeline.shap_prediction(
                pred_df
            )

            # Top 10 Important Features
            shap_table = shap_df.head(10).to_html(

                classes="table table-bordered table-striped",

                index=False
            )

            # ==================================================
            # Render Template
            # ==================================================

            return render_template(

                'home.html',

                results=predicted_price,

                shap_table=shap_table
            )

        except Exception as e:

            print("Error:", e)

            return render_template(

                'home.html',

                results="Prediction Error Occurred"
            )


# ======================================================
# Main Function
# ======================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True
    )