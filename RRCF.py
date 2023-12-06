#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
import rrcf
from gen_toyData import DataFrameGenerator

num_rows = 5000
frequency = 1/1300
frequency_std = 1/1000
ratio_accidental = 0.001
n_features = 4

generator = DataFrameGenerator(num_rows = num_rows,
                               frequency = frequency,
                               frequency_std = frequency_std,
                               ratio_accidental = ratio_accidental,
                               n_features = n_features)
# generator.add_accidental_high_values()
df = generator.gen_df()
plt.figure(figsize=(12,8))
plt.plot(df[['Feature_1']])
plt.grid()
plt.show()
X = df['Feature_1'].astype(float).values

num_trees = 200
shingle_size = 4 #48
tree_size = 1000

points = rrcf.shingle(X, size=shingle_size)
points = np.vstack([point for point in points])
n = points.shape[0]
sample_size_range = (n // tree_size, tree_size)

forest = []
while len(forest) < num_trees:
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    trees = [rrcf.RCTree(points[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)

avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)

for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
    
avg_codisp /= index
avg_codisp.index = df.iloc[(shingle_size-1):].index

############# Isolation Forest for the comparison
contamination = ratio_accidental
IF = IsolationForest(n_estimators=num_trees,
                     contamination=contamination,
                     max_samples='auto',
                     random_state=0)
IF.fit(df[['Feature_1']])
if_scores = IF.decision_function(df[['Feature_1']])*-1
if_anomaly = IF.predict(df[['Feature_1']])
df['if_anomaly'] = if_anomaly
#############

avg_codisp = ((avg_codisp - avg_codisp.min()) / (avg_codisp.max() - avg_codisp.min()))
if_scores = ((if_scores - if_scores.min()) / (if_scores.max() - if_scores.min()))

df['rrcf_scores'] = avg_codisp
df['if_scores'] = if_scores

score_mean = df['rrcf_scores'].mean()
score_std = df['rrcf_scores'].std()
score_based_threshold = score_mean + 3 * score_std
over_threshold = df['rrcf_scores'] > score_based_threshold


fig, ax = plt.subplots(2, figsize=(12,8))
ax[0].plot(df['Feature_1'], color='0.5', label = 'Pseudo Data')
ax[0].scatter(df.index[df['if_anomaly'] == -1], df['Feature_1'][df['if_anomaly'] == -1], color='springgreen', label='Anomaly [IF]')
ax[0].scatter(df.index[over_threshold], df['Feature_1'][over_threshold], color='red', marker='*', alpha=0.6, label='Anomaly [RRCF]')
ax[1].plot(df['if_scores'], color='#7EBDE6', alpha=0.8, label='Score [IF]')
ax[1].plot(df['rrcf_scores'], color='#E8685D', alpha=0.8, label='Score [RRCF]')

ax[0].legend(frameon=True, loc=2, fontsize=12)
ax[0].set_ylabel('Pseudo data Test', size=13)
ax[0].set_title('Anomaly detection Test', size=14)
ax[1].legend(frameon=True, loc=2, fontsize=12)
ax[1].set_ylabel('Normalized Anomaly Score', size=13)

ax[0].xaxis.set_ticklabels([])
ax[0].grid()
ax[1].grid()
ax[0].set_xlim(df.index[0], df.index[-1])
ax[1].set_xlim(df.index[0], df.index[-1])
plt.tight_layout()

plt.show()