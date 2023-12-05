#### This part is making the dataframe sample for the function test.

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_blobs
import math


class DataFrameGenerator:
    def __init__(self, num_rows:int, frequency:float, frequency_std:float, ratio_accidental:float, n_features:int):
        self.num_rows = num_rows
        self.frequency = frequency
        self.frequency_std = frequency_std
        self.ratio_accidental = ratio_accidental
        self.n_features = n_features
        self.df = self._initialize_df()

    @staticmethod
    def elastic_naming(form:str, n:int):
        return [f"{form}_{i}" for i in range(n)]
    

    def calRandomValues(self, seq:int):
        pedestal = 30 + 10 * math.sin(math.pi * self.frequency * seq)
        mean_val = pedestal + 10 * math.sin(2 * math.pi * self.frequency * seq)
        std_val = 2 + math.cos(2 * math.pi * self.frequency_std * seq)
        random_values = np.random.normal(loc=mean_val, scale=std_val, size=(1, 4))
        return random_values

    def _initialize_df(self, start_date:str="2023-01-01"):
        date_range = pd.date_range(start=start_date, periods=self.num_rows, freq='S')
        columns = self.elastic_naming('Feature', self.n_features)
        df = pd.DataFrame(index=date_range, columns=columns)
        return df

    def gen_df(self, start_date:str="2023-01-01"):
        for i in range(self.num_rows):
            random_values = self.calRandomValues(i)
            self.df.iloc[i] = random_values

        self.df = self.add_accidental_high_values(self.df)
        return self.df.clip(0, 100)

    def add_accidental_high_values(self, df:pd.DataFrame):
        num_high_values = int(self.num_rows * self.ratio_accidental)
        high_value_indices = np.random.choice(self.num_rows, num_high_values, replace=False)

        for col in df.columns:
            high_values = np.random.randint(65, 99, size=num_high_values)
            df.loc[df.index[high_value_indices], col] = high_values
        return df

    def gen_testblobs_df(self, n_samples:int, n_features:int):
        X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=42)
        column_names = self.elastic_naming('Feature', n_features)
        df = pd.DataFrame(X, columns=column_names)
        scaler = StandardScaler()
        return scaler.fit_transform(df)
