# More of this on intermediate python
import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
df = pd.read_csv('sales.csv', index_col = 'month')
df


# 1. INDEX
#### USE SQUARE BRACKETS
# Sale of salt in Jan
df['salt']['Jan']


#### USE COLUMN ATTRIBUTE AND ROW LABEL
# Sale of eggs in Mar
df.eggs['Mar']


#### USE .LOC ACCESSOR
# Spam in May
df.loc['May', 'spam']

#### USE .ILOC ACCESSOR
# Spam in May
df.iloc[4,2]


#### SELECT ONLY SOME COLUMNS
# Sale in salt and eggs
df[['salt', 'eggs']]


# 2. SLICE
#### USE SQUARE BRACKET
# Eggs and spam from Feb to Apr
df[['eggs', 'spam']][1:4]


#### USE .LOC
# All rows, some columns
df.loc[:, 'eggs':'salt']

# Some rows, all columns
df.loc['Jan':'Apr', :]

# Some rows, some columns
df.loc['Mar':'May', 'salt':'spam']

# Can use list rather than slice
df.loc['Mar':'May', ['salt','spam']]

# SLice in reverse order
df.loc['May':'Mar':-1]

#### USE .ILOC
df.iloc[2:5, 1:]


#### SERIES VS 1-COLUMN DATAFRAME
df['eggs']
type(df['eggs'])
df[['eggs']]
type(df[['eggs']])