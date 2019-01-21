# More of this on intermediate python
import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
df = pd.read_csv('sales.csv', index_col = 'month')
df


# FILTER WITH BOOLEAN SERIES
#### BOTH CONDITIONS
df[(df.salt >= 50) & (df.eggs < 200)]

#### EITHER CONDITIONS
df[(df.salt >= 50) | (df.eggs < 200)]


#### CREATE DATAFRAME WITH ZEROS AND NANS
df2 = df.copy()
df2['bacon'] = [0, 0, 50, 60, 70, 80]
df2


# NA DATA
#### SELECT COLUMNS WITH ALL NONZEROS
df2.loc[:, df2.all()]


#### SELECT COLUMNS WITH ANY NONZEROS
df2.loc[:, df2.any()]


#### SELECT COLUMNS WITH ANY NANS
df2.loc[:, df2.isnull().any()]


#### SELECT COLUMNS WITHOUT NANS
df2.loc[:, df2.notnull().all()]


#### DROP ROWS
# Drop rows with any NaNs
df2.dropna(how='any')

# Drop rows with all NaNs
df2.dropna(how='all')


#### MODIFY A COLUMN BASED ON ANOTHER COLUMN
# Increase sale for eggs by 5 in months where salt > 55
df.eggs[df.salt > 55] += 5
