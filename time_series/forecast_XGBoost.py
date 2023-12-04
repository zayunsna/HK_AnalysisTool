#!/usr/bin/env python
# -*- coding: utf-8 -*-
from xgboost import XGBRegressor
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from sklearn.metrics import mean_absolute_percentage_error
import optuna

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math


## Loading the simulated data by 'demo_makeBigDF.py'
csv_path = "Pseudo_data_biggest.csv"
df = pd.read_csv(csv_path)
df.rename(columns ={'Unnamed: 0':'Datetime'}, inplace=True)
# df = df[['Datetime', 'Item1']]

## Select a single column for focusing.
df = df[['Datetime','Item1']][:1000]
print(df.head())
# df = pd.melt(df, id_vars=['Datetime'], var_name='item_type', value_name='value')
# df['Datetime'] = pd.to_datetime(df['Datetime'])

TotalEntry = len(df['Datetime'])
Dividing = 0.7
Threshold = math.ceil(TotalEntry*Dividing)
print(" Total Entries : {}".format(TotalEntry))
print(" Threshold of trainig set : {} %".format(Dividing*100))
train = df[:Threshold]
print(" size of training set : {}".format(len(train['Datetime'])))
train = pd.melt(train, id_vars=['Datetime'], var_name='unique_id', value_name='y')
train = train.rename(columns={'Datetime':'ds'})
train['ds'] = pd.to_datetime(train['ds'])
train = train.sort_values(by='ds')
print(train)

valid = df[Threshold:]
print(" size of validation set : {}".format(len(valid['Datetime'])))
valid = pd.melt(valid, id_vars=['Datetime'], var_name='unique_id', value_name='y')
valid = valid.rename(columns={'Datetime':'ds'})
valid['ds'] = pd.to_datetime(valid['ds'])
valid = valid.sort_values(by='ds')
print(valid)

## Missing value or problem child control if needed.
##


def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.1, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
    
    lags = trial.suggest_int('lags', 80, 120, step=4) # step means we only try multiples of 7 starting from 14

    models = [XGBRegressor(random_state=0,
                           n_estimators=500,
                           learning_rate=learning_rate,
                           max_depth=max_depth,
                           min_child_weight=min_child_weight,
                           subsample=subsample,
                           colsample_bytree=colsample_bytree)]

    model = MLForecast(models=models,
                    freq='S',
                    lags=[10,30,60,lags],
                    lag_transforms={
                       1: [(rolling_mean, 30), (rolling_max, 30), (rolling_min, 30)],
                    }, # removing this is better
                    date_features=['second', 'minute'],
                    num_threads=1)


    model.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
    h = valid['ds'].nunique()
    p = model.predict(h=h)
    p = p.merge(valid[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')

    error = mean_absolute_percentage_error(p['y'], p['XGBRegressor'])
    return error

optimizer = optuna.create_study(direction='minimize')
optimizer.optimize(objective, n_trials=20)

bestParm = optimizer.best_params
bestVal = optimizer.best_value
print("%"*50)
print("Best Score:", optimizer.best_value)
print("Best trial:", optimizer.best_trial.params)
print("%"*50)

plot_optimization_history(optimizer)
plot_contour(optimizer)
plot_param_importances(optimizer)
plot_edf(optimizer)
plot_intermediate_values(optimizer)
plot_parallel_coordinate(optimizer)
plot_rank(optimizer)
plot_slice(optimizer)
plot_timeline(optimizer)


## Define the XGBoost model
models = [XGBRegressor(random_state=0,
                        n_estimators=500,
                        learning_rate=bestParm['learning_rate'],
                        max_depth=bestParm['max_depth'],
                        min_child_weight=bestParm['min_child_weight'],
                        subsample=bestParm['subsample'],
                        colsample_bytree=bestParm['colsample_bytree'])]

## Create instance of model
model = MLForecast(models=models,
                   freq='S',
                   lags=[bestParm['lags']],
                   lag_transforms={
                       1: [(rolling_mean, 30), (rolling_max, 30), (rolling_min, 30)],
                   },
                   date_features=['second','minute'],
                   num_threads=1
                   )

model.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])

h = valid['ds'].nunique()
p = model.predict(h=h)
print(p)
p = p.merge(valid[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
print(p)

error = mean_absolute_percentage_error(p['y'], p['XGBRegressor'])

print(error)

plt.figure(figsize=(14,8))

plt.plot(p['y'], 'b-', label='Original data')
plt.plot(p['XGBRegressor'], 'r-', label='XGBoost Result')
plt.grid()
plt.legend()
plt.show()
