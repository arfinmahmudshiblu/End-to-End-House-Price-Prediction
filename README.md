# End-to-End-House-Price-Prediction

# Objective

- To predict housing prices in Ames, Iowa using the "Ames Housing Price Dataset."
- To develop a predictive model with high accuracy and deploy the model as an interactive web application using Flask.


# Introduction

- The Ames Housing Price Dataset is a well-known dataset in the data science community, often used for regression tasks. This project aims to build a predictive model to estimate housing prices based on various features of the houses. After achieving satisfactory results, the model was deployed on the Flask platform to create an interactive web application.

# Dataset Overview

- Rows: 2,930 houses
- Columns: 80+ features
- Target Variable: SalePrice
- Task Type: Regression
- Location: Ames, Iowa, USA

# Common Features
- House size
- Number of rooms
- Basement area
- Garage size
- Neighborhood
- Year built
- Lot area
- Overall quality

# Typical End-to-End Workflow
1. Load and understand data
2. Perform EDA (Exploratory Data Analysis)
3. Handle missing values
4. Encode categorical variables
5. Feature engineering
6. Split train/test data
7. Train regression models
8. Evaluate using MAE/RMSE/R²
9. Generate predictions
10. SHAP Value Prediction


# Popular Models
- Linear Regression
- Decision Tree
- Random Forest
- Ridge Regression
- Lasso Regression
- XGBoost
- Support Vector Machine
- LightGBM

# Regressor
- Stacking Regressor
- Voting Regressor

# Common Challenges
- Missing values
- Categorical encoding
- Outliers
- Feature selection

# Conclusion
- This project successfully built and deployed a predictive model for estimating housing prices in Ames, Iowa. The model provides accurate predictions and is accessible through a user-friendly Flask application. The insights gained from this project can be applied to similar real estate prediction tasks.

# Tools and Technologies
- EDA and Machine Learning: Pandas, Numpy, Matplotlib, Seaborn, SKLearn, Flask, pickle, dill, xgboost, lightgbm, shap, dump.
- Testing and Deployment: Fask.

# Setup
- Set up a virtual environment using virtualenv or conda and add Python>=3.12.12 as your interpreter. If using conda then perform conda install pip for effective package management.
- Add dependencies using pip install -r requirements.txt
- Move into directory path which contains the file app.py. Run the app using flask run app.py

# Acknowledgments
- This project was completed as part of an independent study on machine learning and model deployment. Special thanks to the data science community for providing the Ames Housing Price Dataset and to the developers of Flask for their robust and accessible platform.

# References
- Ames Housing Dataset: Ames Housing Dataset on Kaggle.


## How to run?

```bash
git clone https://github.com/arfinmahmudshiblu/End-to-End-House-Price-Prediction.git
```


```bash
conda create -n webapp python=3.12 -y
```

```bash
conda activate webapp
```

```bash
pip install -r requirements.txt
```

```bash
python -m src.components.data_ingestion
```

