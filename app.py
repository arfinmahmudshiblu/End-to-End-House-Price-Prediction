from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:

        data = CustomData(
            overall_qual=int(request.form.get("Overall Qual")),
            overall_cond=int(request.form.get("Overall Cond")),
            total_bsmt_sf=float(request.form.get("Total Bsmt SF")),
            first_flr_sf=float(request.form.get("1st Flr SF")),
            gr_liv_area=float(request.form.get("Gr Liv Area")),
            garage_area=float(request.form.get("Garage Area")),
            ms_zoning=request.form.get("MS Zoning"),
            neighborhood=request.form.get("Neighborhood"),
            house_style=request.form.get("House Style"),
            heating_qc=request.form.get("Heating QC"),
            central_air=request.form.get("Central Air"),
            kitchen_qual=request.form.get("Kitchen Qual"),
            garage_type=request.form.get("Garage Type"),
            sale_condition=request.form.get("Sale Condition"),
        )

        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template(
            "home.html",
            results=round(results[0], 2)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)