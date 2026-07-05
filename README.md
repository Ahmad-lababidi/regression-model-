# regression-model-

# Vehicle Price Prediction Pipeline

## 📌 Project Overview

This repository contains a comprehensive machine learning pipeline designed to predict used car prices. All data processing, feature engineering, and model training logic are contained within the `"movie_2.ipynb"` file. The project applies various regression techniques to both a manufacturer-specific dataset and a broader, general vehicles dataset.

## 📊 Data Preprocessing & Feature Engineering

The `"movie_2.ipynb"` notebook implements a robust data cleaning pipeline to handle missing values, correct entry errors, and prepare features for the machine learning algorithms.

* **Audi Dataset Preparation:**
* The script removes duplicate rows to prevent model bias.


* Samples containing a `0.0` engine size are dropped as data entry errors.


* The cleaned subset is exported as `audiCleaned.csv`.




* **General Vehicles Dataset Cleaning:**
* Samples with missing manufacturer, model, or year data are dropped.


* Missing values in the condition, cylinders, fuel, and drive columns are filled with an 'unknown' placeholder.


* The data is aggressively filtered to remove outliers: prices are restricted to between $500 and $100,000, model years must be post-1990, and odometer readings must be below 300,000.




* **Feature Engineering:**
* A continuous `age` feature is calculated by subtracting the vehicle's manufacturing year from a baseline year of 2025.


* A `manufacturerRegion` feature is created to group brands into broad geographic categories (USA, Japan, Germany, Korea, Europe, and Other).


* String-based vehicle conditions are mapped to a numeric `conditionRank` scale ranging from 0 (salvage) to 5 (new).


* Cylinder counts are parsed and cleaned into standard integer formats.


* A `model_freq` column is calculated based on the frequency counts of specific car models.




* **Target Normalization:**
* To account for a right-skewed raw price distribution, the target variable undergoes a log transformation using `np.log1p` to normalize the data.





## 🧠 Modeling Architecture

The modeling strategy incorporates strict feature scaling and tests several regressors to find the optimal fit.

* **Data Splitting:** Data is separated into an 80% training set and a 20% testing set.


* **Scaling & Encoding:** Numeric columns are normalized using `StandardScaler`. Categorical variables are converted to numeric formats via one-hot encoding (`pd.get_dummies`), and the test set is reindexed to align perfectly with the training features.


* **Algorithms Evaluated:** The pipeline tests Support Vector Regression (SVR), Stochastic Gradient Descent (SGDRegressor), Random Forest Regressor, and Gradient Boosting Regressor.



## 📈 Evaluation & Results

To ensure metrics are interpretable, log-predicted prices are inversely transformed back to their real-world dollar values using `np.expm1`. The models are evaluated on Root Mean Squared Error (RMSE), R-squared (R2), and Mean Absolute Error (MAE).

* **Audi Dataset Highlights:**
* The baseline Random Forest model achieved excellent results with an R2 Score of 0.9573 and an MAE of $1,633.13.


* Support Vector Regression utilizing an RBF kernel also performed highly, resulting in an R2 Score of 0.9453.




* **General Vehicles Dataset Highlights:**
* An Optimized Random Forest model (configured with 200 estimators, a minimum of 2 samples per leaf, and 'sqrt' max features) handled the high-dimensional data best, achieving an R2 Score of 0.8683 and an RMSE of $5,125.53.


* The Linear SGD-SVR model struggled to capture the variance in the broader dataset, yielding an R2 Score of 0.6533 and an RMSE of $8,316.15.
