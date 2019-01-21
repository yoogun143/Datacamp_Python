# More of this on intermediate python
import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
df = pd.read_csv('sales.csv', index_col = 'month')
df


#### USE DATAFRAME VECTORIZED METHODS
df['dozens_of_eggs'] = df.eggs.floordiv(12) # Convert to dozens unit for eggs
df

#### USE NUMPY VECTORIZED FUNCTIONS (fastest)
import numpy as np
np.floor_divide(df, 12) # Convert to dozens unit


#### USE PLAIN PYTHON FUNCTIONS
def dozens(n):
    return n//12
df.apply(dozens) # Convert to dozens unit

df.apply(lambda n: n//12)


#### TRANSFORM INDEX
# Index to upper
df.index = df.index.str.upper()
df

# Index to lower
df.index = df.index.map(str.lower)
df

# Upper again: (part 2)
df.index = [month.upper() for month in df.index]
df

# Change index name label (part 2)
df.index.name = 'MONTHS'
df.columns.name = 'PRODUCTS'
df


#### MUTATE NEW COLUMN USING OTHER COLUMNS
df['salty_eggs'] = df.salt + df.dozens_of_eggs
df


#### USE MAP WITH A DICTIONARY
election = pd.read_csv('pennsylvania2012_turnout.csv')
# Create the dictionary: red_vs_blue
red_vs_blue = {"Obama": "blue", "Romney": "red"}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election.winner.map(red_vs_blue)

# Print the output of election.head()
print(election.head())
