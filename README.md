# Used Car Price Prediction

## Project Overview

This repository contains a machine learning pipeline for predicting used-car prices using an Audi dataset and a broader vehicles dataset. The notebook is [`car price model.ipynb`](car%20price%20model.ipynb).

The workflow cleans and prepares both datasets, engineers features, log-transforms price targets, and compares Support Vector Regression (SVR), SGD Regressor, Random Forest, and Gradient Boosting models.

## Data Preprocessing and Feature Engineering

The notebook removes duplicate Audi rows and records invalid zero engine sizes. For the general vehicles data, it filters out incomplete records and outliers, fills selected categorical missing values, and creates features including vehicle age, manufacturer region, condition rank, cylinder count, and model frequency.

Price targets are log-transformed with `numpy.log1p`. Numeric features are scaled with `StandardScaler`, and categorical features are one-hot encoded with `pandas.get_dummies`.

## Models and Results

The notebook evaluates SVR, SGD Regressor, Random Forest Regressor, and Gradient Boosting Regressor. On the Audi data, the Random Forest model achieved an R² of 0.9573 and an MAE of $1,633.13; SVR achieved an R² of 0.9453. On the general vehicles data, the optimized Random Forest model achieved an R² of 0.8683.

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Ahmad-lababidi/used-car-price-prediction.git
   cd used-car-price-prediction
   pip install -r requirements.txt
   ```
2. Download the Audi dataset (`audi.csv`) and general vehicles dataset (`vehicles.csv`) from the Kaggle sources used for this project, and place them in the working folder expected by the notebook. The original Kaggle dataset URLs were not available, so confirm the correct source links before sharing this README.
3. Open [`car price model.ipynb`](car%20price%20model.ipynb) and run all cells. Update the notebook's local `os.chdir` and CSV paths to match your computer before running.
