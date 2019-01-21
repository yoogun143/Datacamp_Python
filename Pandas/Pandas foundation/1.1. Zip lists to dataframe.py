import os
os.chdir('E:\Datacamp\Python\Pandas\Pandas foundation')
import pandas as pd
df = pd.read_csv('world_population.csv')
list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]] 


#### ZIP LIST TO DATAFRAME
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)
