import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

## Function for chainging normal value into accidental events
def change_components(df:pd.DataFrame, row, col, new):
    row_size = len(row)
    col_size = len(col)
    for i in range(col_size):
        for j in range(row_size):
            df.iloc[row[j], df.columns.get_loc(col[i])] = new[j]
    return df

num_rows = 50000  # Example size of the DataFrame, can be adjusted as needed
frequency = 1/13000 # frequency of mean value fluctuation.
frequency_std = 1/10000 # frequency of standard deviation variation.

# Create a datetime index with 1-second intervals
date_range = pd.date_range(start="2023-01-01", periods=num_rows, freq='S')

# Create a DataFrame with 4 columns
# The basic values will follow a Gaussian distribution centered around 30
# but the mean should follow the sine wave in order to make a variation.
# and also the std will have cosine wave.
# With a low probability, some data points will reach higher values like between 65 and 99
columns = ['Item1', 'Item2', 'Item3', 'Item4']

df = pd.DataFrame(index=date_range, columns=columns)

for i in range(num_rows):
    mean = 30
    pedestal = mean + 10 * math.sin( math.pi * frequency * i)
    mean_val = pedestal + 10 * math.sin(2 * math.pi * frequency * i)
    std_val = 2 + math.cos(2 * math.pi * frequency_std * i)
    random_values = np.random.normal(loc=mean_val, scale=std_val, size=(1, 4))
    df.iloc[i] = random_values

# Introducing accidental high values
ratio_accitental = 0.0001 ## Probability of accidental events.
num_high_values = int(num_rows * ratio_accitental)
high_value_indices = np.random.choice(num_rows, num_high_values, replace=False)

# switch the accidental high values with normal values.
for col in range(4):
    high_values = np.random.randint(65, 99, size=num_high_values)
    # print("col {} : value {}".format(col, high_values))
    df = change_components(df, high_value_indices, columns, high_values)

df = df.clip(0, 100)
# Save the demo data.
df.to_csv("Pseudo_data_biggest.csv")

plt.figure(figsize=(12,6))
plt.plot(df['Item1'], 'b-')
plt.xlabel('Datetime')
plt.ylabel('Arb.')
plt.grid()
plt.show()
