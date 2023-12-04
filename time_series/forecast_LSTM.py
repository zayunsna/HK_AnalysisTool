#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Sequential, models
from keras.utils import plot_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import math

from alive_progress import alive_bar

SEED = 13
tf.random.set_seed(SEED)
# tf.config.set_visible_devices([], 'GPU')
############################################################
print(f"Tensorflow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"System Version: {sys.version}")

mpl.rcParams['figure.figsize'] = (17, 5)
mpl.rcParams['axes.grid'] = False
sns.set_style("whitegrid")
############################################################


## Apply basic value normalizaing.
## Use MinMaxScaler from Scikit-learn
def applyScaling(data:pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=df.columns)
    return scaler, scaled_data

## In order to restored the data from normalization.
def applyInverseScaling(target:str, data:pd.DataFrame, scaler):
    if target not in scaler.get_feature_names_out():
        raise ValueError("Target column `{}` is not in the scaler`s features.".format(target))
    if target not in data.columns:
        raise ValueError("Target column `{}` is not in the DataFrame".format(target))
    column_data = data[[target]].values
    restored_data = scaler.inverse_transform(column_data)[:, 0]
    return pd.DataFrame(restored_data, index=data.index, columns=[target+'_inversed'])

## Build the data structure for training & Validating.
def setDataStructure(dataset:np.ndarray, target:np.ndarray, start_idx:int, end_idx:int,
                     history_size:int, target_size:int, step:int, single_step:bool=False):
    data=[]
    labels=[]
    start_idx = start_idx+history_size
    if end_idx is None:
        end_idx = len(dataset)-target_size

    for i in range(start_idx, end_idx):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

## Function for a Multiplt data point prediction.
## Not only the single point forecasting, but also multiple data point can be forecasted by given 'steps'
def iterative_forecast(model, start_data, steps):
    forecast = []
    current_data = start_data.copy()  # Copy to prevent modifying original data

    # Ensure current_data is 2D: [time_steps, features]
    if current_data.ndim == 1:
        current_data = current_data.reshape(-1, 1)

    with alive_bar(steps) as bar:
        for _ in range(steps):
            # Reshape to 3D: [batch_size, time_steps, features]
            current_batch = current_data.reshape(1, -1, 1)

            prediction = model.predict(current_batch, verbose=0)[0, 0]  # Predict next step
            forecast.append(prediction)

            # Update current_data for next prediction
            current_data = np.roll(current_data, -1)
            current_data[-1] = prediction
            bar()

    return np.array(forecast)

## Create the timestape for the forecating results.
def createTimeStamp(length:int):
    return list(range(-length, 0))

## Make data frame for the forecasting results.
## this function will coaa the 'createTimeStampe
def makeDataFrame(npa:np.ndarray, target:str, lastDate:str, timeUnit:str, scaler):
    dataframe = pd.DataFrame(npa, columns=[target])
    print(dataframe.head())
    restored_data = applyInverseScaling(target, dataframe, scaler)
    print(restored_data.head())
    datetime_frame = pd.DataFrame(pd.date_range(lastDate, freq=timeUnit, periods=len(restored_data)))
    result_data = pd.concat([datetime_frame, restored_data], axis=1)
    return result_data

## Draw the result with true data and forecated data.
def drawResultPlot(history, true_future, prediction):
    plt.figure(figsize=(18,6))
    num_in = createTimeStamp(len(history))
    num_out = len(prediction) if prediction is not None else 0

    plt.plot(num_in, np.array(history), label='History')
    
    if true_future is not None:
        plt.plot(np.arange(len(true_future)), np.array(true_future), 'b', label='True Future')
        
    if prediction is not None:
        plt.plot(np.arange(num_out), np.array(prediction), 'r^', label='Predicted Future')
    
    plt.legend()
    plt.show()


## Loading the simulated data by 'demo_makeBigDF.py'
csv_path = "Pseudo_data_biggest.csv"
df = pd.read_csv(csv_path)
df.rename(columns ={'Unnamed: 0':'Datetime'}, inplace=True)

## keep the lasted data for the stating point of prediction.
lastDate = df[['Datetime']][-1:].values[0][0]

## Select the target column.
target_name = 'Item1'
df = df[[target_name]]
print(df.head())
print(" Size of Dataset : {}".format(len(df)))
print("#"*100)

## Applying the scaling.
scaler, scaled_df = applyScaling(df)
scaled_df = scaled_df.values
print(scaled_df[:, 0])
print("#"*100)

## Set Parameters
BATCH_SIZE = 32 #128
BUFFER_SIZE = 10000
TOTAL_DATASET = df.size
TRAIN_SPLIT = math.ceil(TOTAL_DATASET*0.7) # Split row for training.

EVALUATION_INTERVAL = 200
EPOCHS = 50
PATIENCE = 5
past_history = 5000
future_target = 1
STEP = 10

print(type(df))
print(type(scaled_df))
print(type(scaled_df[:, 0]))
print("#"*100)

## Build the training & Validation set based on defined parameters.
x_train, y_train = setDataStructure(dataset=scaled_df, 
                                    target=scaled_df[:,0], 
                                    start_idx=0, 
                                    end_idx=TRAIN_SPLIT, 
                                    history_size=past_history, 
                                    target_size=future_target, 
                                    step=STEP,
                                    single_step=True)

x_valid, y_valid = setDataStructure(dataset=scaled_df, 
                                    target=scaled_df[:,0], 
                                    start_idx=TRAIN_SPLIT, 
                                    end_idx=None, 
                                    history_size=past_history, 
                                    target_size=future_target, 
                                    step=STEP,
                                    single_step=True)

## Slicing each dataset.
## Purpose : to fit the input shape of LSTM 
## - from_tensor_slices : make a tensor bulk.
## - batch : seperate the dataset into mini-batch's using given parameter.
## - repeat : In order to match the dataset and EPOCH, will repeat batch creation.
## - shuffle : iterelly shuffle the mini-batch's.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = validation_dataset.batch(BATCH_SIZE).repeat()

## defined the model name to saving.
model_path = "model_verBiggest.h5"

## set a training status.
## True : model training is processed.
## False : Load the model and directly go the forecasting.
do_training = False


if do_training: ## Train LSTM model
    input_shape = x_train.shape[-2:]
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(16, activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.summary())

    early_stopping = [ EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)]

    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=validation_dataset,
                        validation_steps=EVALUATION_INTERVAL,
                        callbacks=[early_stopping])
else : ## Load LSTM Model without model training.
    model = models.load_model(model_path)


## Set-up the forecast data.
forecast_steps = 100 ## the number of forecast data point.
history_length = 500 ## the number of true data point for drawing.
last_known_data = scaled_df[-history_length:, 0]
forecast = iterative_forecast(model, last_known_data, forecast_steps)

## Draw the result plot with original and forecasting data.
drawResultPlot(last_known_data, None, forecast)

## Save the forecasting results into csv file
forecast = makeDataFrame(forecast, target_name, lastDate, '1S', scaler)
forecast.columns = ['Datetime',target_name+'_forecast']  
forecast.to_csv('forecast_output.csv')



