#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import array

################################################################################################################################
#### This part is making the dataframe sample for the function test.
import random
import string
# Define the number of rows and columns
n_rows = 1000
n_cols = 8

# Create random data for each column
data = {
    'Object_String': ["".join(random.choices(string.ascii_letters, k=8)) for _ in range(n_rows)],
    'Object_Bool': [random.choice([True, False]) for _ in range(n_rows)],
    'Float_1': np.random.logistic(1, 1, n_rows),
    'Float_2': np.random.normal(5, 1, n_rows),
    'Int': np.random.randint(1, 100, size=(n_rows)),
    'Float_3': np.random.gamma(2., 2., n_rows),
    'Date': pd.date_range(start='2023-01-01', periods=n_rows, freq='D'),
    'Mixed': [random.choice([random.randint(1, 50), random.uniform(1.0, 50.0), 
                             "".join(random.choices(string.ascii_letters, k=5))]) for _ in range(n_rows)]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['Int'] = df['Int'].astype('Int64')

# Introduce approx 5~10% missing values
for col in df.columns:
    frac = random.randrange(5,10) * 0.01
    df.loc[df.sample(frac=frac).index, col] = np.nan
#### sample dataframe part end.
################################################################################################################################

class DataAnalysisTool:
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def create_grid(n):
        root = n ** 0.5
        rows = int(root) if root.is_integer() else int(root) + 1
        cols = rows if rows * (rows - 1) < n else rows - 1
        return rows, cols

    def get_info(self):
        print("\n### Data Summary")
        print(self.dataset.describe())
        print("\n### Summarized data")
        print(self.dataset.info())
        print("\n### Data Head info")
        print(self.dataset.head())

    def columns_info(self, types):
        column = self.dataset.select_dtypes(include=types)
        column_name = column.columns
        column_count = len(column_name)
        return column_name, column_count

    def draw_plot(self, plot_type, grid_x, grid_y, entry, items, nbins=50, ylog=False):
        fig, ax = plt.subplots(grid_x, grid_y, figsize=(10, 10))
        ax = ax.flatten()
        for i in range(entry):
            ax[i].grid()
            if ylog: 
                ax[i].set_yscale('log')
            if plot_type == 'num': 
                sns.histplot(self.dataset[items[i]], color='b', bins=nbins, ax=ax[i])
            else: 
                sns.countplot(x=self.dataset[items[i]], ax=ax[i])
        plt.show()

    def numerical_plot(self, nbins=50, ylog=False):
        numerical_column_name, column_count = self.columns_info(['int', 'float'])
        grid_x, grid_y = DataAnalysisTool.create_grid(column_count)
        self.draw_plot("num", grid_x, grid_y, column_count, numerical_column_name, nbins, ylog)

    def categorical_plot(self):
        categorical_column_name, column_count = self.columns_info(['object'])
        grid_x, grid_y = DataAnalysisTool.create_grid(column_count)
        self.draw_plot("category", grid_x, grid_y, column_count, categorical_column_name)

    @staticmethod
    def scaler(data):
        sc = MinMaxScaler(feature_range=(0, 1))
        sc = sc.fit(data)
        scaled_data = sc.transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

    @staticmethod
    def cal_pcc(feature_a, feature_b, _range):
        fitter, C_p = np.polyfit(feature_a, feature_b, 1, cov=True)
        
        xModel = np.linspace(_range[0][0], _range[0][1], 500)
        yModel = np.polyval(fitter, xModel)

        TT = np.vstack([xModel**(1-i) for i in range(2)]).T
        yi = np.dot(TT, fitter)
        C_yi = np.dot(TT, np.dot(C_p, TT.T))
        sig_yi = np.sqrt(np.diag(C_yi))

        return xModel, yModel, yi, sig_yi

    def get_feature_correlation(self, feature_a, feature_b, _range):
        xData = self.dataset[feature_a]
        yData = self.dataset[feature_b]
        xModel, yModel, yi, sig_yi = DataAnalysisTool.cal_pcc(xData, yData, _range)

        fig = plt.figure(figsize=(8, 6), dpi=100)
        fig = fig.add_subplot(111)

        plt.hist2d(x=xData, y=yData, bins=50, norm=mpl.colors.LogNorm(), range=_range)
        fig.plot(xModel, yModel)
        fig.fill_between(xModel, yi+sig_yi, yi-sig_yi, alpha=.25)
        plt.colorbar()
        plt.grid()
        plt.xlabel(feature_a)
        plt.ylabel(feature_b)
        plt.show()

    def get_feature_heatmap(self):
        target_column_name, _ = self.columns_info(['int', 'float'])
        numerical_dataset = self.dataset[target_column_name]
        scaled_numerical_dataset = DataAnalysisTool.scaler(numerical_dataset).corr()
        plt.figure(figsize=(14, 10))
        sns.heatmap(scaled_numerical_dataset, cmap='Blues', vmin=-1, vmax=1, annot=True)
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def remove_columns(self, column_names):
        for name in column_names:
            self.dataset = self.dataset.drop(name, axis=1)
        return self.dataset

# Example usage
# df = numpy.random # Test data

# data_tool = DataAnalysisTool(df)
# data_tool.get_info()
# data_tool.numerical_plot()
# data_tool.categorical_plot()
# data_tool.get_feature_heatmap()
# data_tool.remove_columns(['column1', 'column2'])
