# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 10:48:16 2021
my implementation of the functionality found in
https://medium.com/@oemer.aslantas/a-real-world-example-of-predicting-sales-volume-using-xgboost-with-gridsearch-on-a-jupyternotebook-c6587506128d


@author: Josh.Galloway
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (15,10)
pd.set_option('precision', 3)
pd.set_option('display.max_columns',50)
np.set_printoptions(precision=3)
warnings.filterwarnings('ignore')
  
import seaborn as sns
import datetime as dt
import xgboost as xgb

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from german_holidays import get_german_holiday_calendar



''' Prepare the data
'''

# read in data
df = pd.read_excel('SalesData.xlsx')
df.info()
df.head()
# data is date and time of sale for number of units (only two columns)

# group total sales by day
df['Date'] = df.From.apply(lambda x: x.date())
df.head()
totals_df = pd.pivot_table(df,index=['Date'],values=['Sold Units'], aggfunc=np.sum)
totals_df.reset_index(inplace=True)
totals_df.head()

# check description
totals_df['Sold Units'].describe()
# plot
totals_df.plot(marker='o', linestyle='none')

'''
Feature Engineering
'''
# author of tutorial removes holidays but I think it's better to 
#... create a variable to indicate them and leave in 

# import german holidays
cal_cls = get_german_holiday_calendar('NW')
cal = cal_cls()
holidays = [
    h.date() for h in pd.to_datetime(cal.holidays(start='2012',end='2020'))
    ]
totals_df['is_holiday'] = totals_df.Date.apply(lambda x: x in holidays)
totals_df.is_holiday = totals_df.is_holiday.astype(int)

# add more features like in tutorial
# parse out year day and month
totals_df['year'] = pd.to_datetime(totals_df.Date).dt.year
totals_df['month'] = pd.to_datetime(totals_df.Date).dt.month
totals_df['day'] = pd.to_datetime(totals_df.Date).dt.day
totals_df['day_of_week'] = pd.to_datetime(totals_df.Date).dt.dayofweek
totals_df['is_weekend'] = totals_df.day_of_week > 4
totals_df.is_weekend = totals_df.is_weekend.astype(int)

totals_df.head(10)

'''
EDA
'''
# check correlation matrix
totals_df.corr()

# distribution of sold units
sns.histplot(data = totals_df['Sold Units'])
# trimodal mess...
# look at it without holidays
sns.histplot(data = totals_df[totals_df.is_holiday == 0]['Sold Units'])
# same...
# truncate date range to everything from 2016 on like author
totals_df = totals_df[totals_df.Date > dt.datetime(2015,12,31).date()]
sns.histplot(data = totals_df[totals_df.is_holiday == 0]['Sold Units'])
sns.boxplot(totals_df['Sold Units'])
# much better...

# top and tail sales at 5 and 95 qunatiles
cap = totals_df['Sold Units'].quantile(0.95)
cut = totals_df['Sold Units'].quantile(0.05)
totals_df = totals_df[totals_df['Sold Units'] < cap]
totals_df = totals_df[totals_df['Sold Units'] > cut]


'''
Test Train Split
'''
# separate most recent part of dataset as test and rest as train
test_size = 0.1
split_date = totals_df.Date.iloc[int((1-test_size)*len(totals_df.index))]

df_test = totals_df[totals_df.Date > split_date].copy(deep = True)
df_train = totals_df[totals_df.Date <= split_date].copy(deep = True)

y_test = df_test['Sold Units'].values
X_test = df_test.drop(['Sold Units','Date'], axis = 1)
y_train = df_train['Sold Units'].values
X_train = df_train.drop(['Sold Units','Date'], axis = 1)

'''
Train XGBoost Model
'''
# grid search the parameter space
params = {
    'min_child_weight': [4,5],
    'gamma': [i/10.0 for i in range(3,6)],
    'subsample': [i/10.0 for i in range(6,11)],
    'colsample_bytree': [i/10.0 for i in range(6,11)],
    'max_depth': [2,3,4]
    }

# Init xgb
xgb_reg = xgb.XGBRegressor(nthread=-1, objective='reg:squarederror')

gs = GridSearchCV(xgb_reg, params,cv=5) 
# start search
gs.fit(X_train, y_train, verbose = 2)

# print best
gs.best_params_
model = gs.best_estimator_
print(model)


'''
Evaluate Model
'''
y_pred = model.predict(X_test)
print(f'R2 Score on Test: {r2_score(y_test,y_pred)}')
print(f'MAE Score on Test: {mean_absolute_error(y_test,y_pred)}')
print(f'RMSE Score on Test: {np.sqrt(mean_squared_error(y_test,y_pred))}')

# examine feature importance
xgb.plot_importance(model)

# plot prediction
df_test['prediction'] = y_pred 
df_results = pd.concat([df_test, df_train], sort=False)
df_results.columns
df_results.set_index('Date',drop=True,inplace=True)


df_results[['Sold Units','prediction']].plot()

# check prediction closer
f,ax = plt.subplots(1)
ax.set_title('Prediciton Results')
_=df_results[['Sold Units','prediction']].plot(
    style=['-','o'], ax = ax)
ax.set_xbound(lower='2019-09-01', upper='2019-10-01')

'''
Save model
'''
from joblib import dump

dump({'model': model,'parameters':gs.best_params_}, 'model_and_params.joblib')

