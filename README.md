# XGBoost Sales Forecasting

This project was created to mimic the functionality outlined in the tutorial found [here](https://medium.com/@oemer.aslantas/a-real-world-example-of-predicting-sales-volume-using-xgboost-with-gridsearch-on-a-jupyternotebook-c6587506128d).

The project comprises data formating, feature engineering, EDA and optimization of an XGBoost regressor via gridsearch.  Overall the forecasting produced very good results with an R<sup>2</sup> of 0.87.

![](https://github.com/jgalloway42/xgboost_sales_forecast/blob/main/Prediction_Results_Close_Up.png)

## Dataset
A data set of sales data for some store was given with 15 minute resolution. The raw file is SalesData.xlsx.

## Data Representation and Processing
The data was convereted to daily total sales and features added for year, month, day, day of week, weekend indicator, and holiday indicator. Following this, EDA was done resulting in the limiting of training data to years including and past 2016 in order to obtain a favorable target distribution which would be most representative of the future sales.  In addition the total sales below the 5% quantile and above the 95% quantile were thrown out. From the remaining data, 10% of the latest sales data was held out as a test set.

![](https://github.com/jgalloway42/xgboost_sales_forecast/blob/main/Raw_Data_Separated_By_Weekend.png)

## Model Derivation
An XGBoost regressor was selected for the model and optimized via gridsearch with 5-fold cross validation over a parameter space including min_child_weight, gamma, subsample, colsample_bytree, and max_depth. These parameters are further explained [here](https://xgboost.readthedocs.io/en/latest/parameter.html). The optimized model parameters where as follows:
| Parameter | Value |
|-----------|-------|
|colsample_bytree|1.0|
|gamma|0.3|
|max_depth|2|
|min_child_weight|5|
|subsample|1.0|

## Results
The feature importances where as follows and as mentioned and shown above the overall the forecasting produced very good results with an R<sup>2</sup> of 0.87. The holidays interestingly didn't play a very large role in the model as I might have expected. The seasonality associated with the day of the month, month and day of the week dominated the model.
![](https://github.com/jgalloway42/xgboost_sales_forecast/blob/main/XGBoost_Model_Feature_Importance.png)
