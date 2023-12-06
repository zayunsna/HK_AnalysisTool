#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_blobs
from gen_toyData import DataFrameGenerator


generator = DataFrameGenerator(num_rows=50000, frequency=1/1300, frequency_std=1/1000, ratio_accidental=0.001, n_features=4)
# generator.add_accidental_high_values()
df = generator.gen_df()

plt.figure(figsize=(12,8))
plt.plot(df[['Feature_1']])
plt.grid()
plt.show()

df = df[['Feature_1']]

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


randseed = np.random.RandomState(42)

# # Splitting the dataset into training and testing sets
# train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# # Isolation Forest
# clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.02), random_state=42)
# clf.fit(train_df[['Feature_1']])

# # Anomaly Scoring
# df['scores'] = clf.decision_function(df[['Feature_1']])
# df['anomaly'] = clf.predict(df[['Feature_1']])


# Fit the model
clf = IsolationForest(contamination=0.001, random_state=randseed)
clf.fit(df[['Feature_1']])

# Predict anomalies
df['scores'] = clf.decision_function(df[['Feature_1']])
df['anomaly'] = clf.predict(df[['Feature_1']])

# Visualize
fig, ax1 = plt.subplots(figsize=(12,8))
ax2 = ax1.twinx()
ax1.plot(df[['Feature_1']], color='blue', label='Data points[line]')
ax2.plot(df[['scores']], color = 'Orange', alpha=0.5, label='Score')
ax1.scatter(df.index, df['Feature_1'], color='k', s=3., label='Data points')
ax1.scatter(df.index[df['anomaly'] == -1], df['Feature_1'][df['anomaly'] == -1], color='red', label='Anomaly')
fig.legend()
plt.show()

# from sklearn.ensemble import IsolationForest
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# # Generating a synthetic dataset
# np.random.seed(42)
# X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1, center_box=(-10.0, 10.0))
# X = np.concatenate([X, np.random.uniform(low=-10, high=10, size=(20, 2))])  # adding some anomalies

# # Applying Isolation Forest
# iso_forest = IsolationForest(contamination=0.06)
# preds = iso_forest.fit_predict(X)

# # Visualizing the results
# plt.figure(figsize=(10, 6))
# plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='coolwarm', marker='o', s=35, edgecolor='k', label='Data Points')
# plt.title('Isolation Forest Anomaly Detection')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Our tiny dataset
# X_tiny = np.array([[5, 7], [6, 9], [8, 8], [9, 6], [15, 15]])

# # Function to build an isolation tree
# def build_isolation_tree(X, depth=0, max_depth=2):
#     n_samples = X.shape[0]
#     if depth == max_depth or n_samples <= 1:
#         return np.array([depth])

#     # Randomly select a feature and a split value
#     feature_idx = np.random.randint(0, X.shape[1])
#     feature_vals = X[:, feature_idx]
#     split_val = np.random.uniform(np.min(feature_vals), np.max(feature_vals))

#     # Split the dataset
#     left_indices = X[:, feature_idx] < split_val
#     right_indices = ~left_indices

#     # Recursively build the left and right branches
#     left_tree = build_isolation_tree(X[left_indices], depth + 1, max_depth)
#     right_tree = build_isolation_tree(X[right_indices], depth + 1, max_depth)

#     # Return the tree structure
#     return np.array([depth, feature_idx, split_val, left_tree, right_tree], dtype=object)

# # Building two simple isolation trees
# tree1 = build_isolation_tree(X_tiny, max_depth=3)
# tree2 = build_isolation_tree(X_tiny, max_depth=3)

# # Displaying the structure of the trees
# print(tree1)
# print(tree2)
