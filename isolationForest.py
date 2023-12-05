# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import pandas as pd
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.datasets import make_blobs
# from gen_toyData import DataFrameGenerator


# generator = DataFrameGenerator(num_rows=5000, frequency=1/1300, frequency_std=1/1000, ratio_accidental=0.001, n_features=4)
# # generator.add_accidental_high_values()
# df = generator.gen_df()

# plt.figure(figsize=(12,8))
# plt.plot(df[['Feature_1']])
# plt.grid()
# plt.show()

# df = df[['Feature_1']]

# from sklearn.ensemble import IsolationForest

# randseed = np.random.RandomState(42)

# # Fit the model
# clf = IsolationForest(contamination=0.05, random_state=randseed)
# clf.fit(df[['Feature_1']])

# # Predict anomalies
# df['scores'] = clf.decision_function(df[['Feature_1']])
# df['anomaly'] = clf.predict(df[['Feature_1']])

# # Visualize
# plt.figure(figsize=(10, 6))
# plt.plot(df[['Feature_1']], color='blue', label='Data points[line]')
# plt.scatter(df.index, df['Feature_1'], color='k', s=3., label='Data points')
# plt.scatter(df.index[df['anomaly'] == -1], df['Feature_1'][df['anomaly'] == -1], color='red', label='Anomaly')
# plt.legend()
# plt.title("Anomaly Detection in Time Series Data")
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.show()

from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generating a synthetic dataset
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1, center_box=(-10.0, 10.0))
X = np.concatenate([X, np.random.uniform(low=-10, high=10, size=(20, 2))])  # adding some anomalies

# Applying Isolation Forest
iso_forest = IsolationForest(contamination=0.06)
preds = iso_forest.fit_predict(X)

# Visualizing the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='coolwarm', marker='o', s=35, edgecolor='k', label='Data Points')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
