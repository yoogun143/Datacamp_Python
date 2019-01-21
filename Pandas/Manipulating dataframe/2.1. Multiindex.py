import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
stocks = pd.read_csv('stocks.txt', sep = ' ')
stocks


#### SET INDEX (SYMBOL AND DATE ARE REPEATED VALUES)
stocks = stocks.set_index(['Symbol', 'Date'])
stocks
stocks.index


#### SORT INDEX
stocks = stocks.sort_index()
stocks


#### INDEX + SLICING
# Index individual row (put into a tuple)
stocks.loc[('CSCO', '2016-10-04')]
stocks.loc[('CSCO', '2016-10-04'), 'Volume']

# Slice outermost index: Symbol
stocks.loc['AAPL']
stocks.loc['CSCO': 'MSFT']
stocks.loc['2016-10-04'] # Cannot do this

# Fancy indexing (outermost index)
stocks.loc[(['AAPL', 'MSFT'], '2016-10-05'), :]
stocks.loc[(['AAPL', 'MSFT'], '2016-10-05'), 'Close']

# Fancy indexing (innermost index)
stocks.loc[('CSCO', ['2016-10-05', '2016-10-03']), :]

# Fancy index: both
stocks.loc[(['AAPL', 'MSFT'], ['2016-10-05', '2016-10-03']), :]

# Slicing both indexes (only want to index innermost level)
stocks.loc[(slice(None), slice('2016-10-03', '2016-10-04')),:]

# INDEX COLUMNS (part 4)
titanic = pd.read_csv('titanic.csv')
# Group titanic by 'pclass': by_class
aggregated = titanic.groupby("pclass")[['age','fare']].agg(["max", "median"])

# Print aggregated
aggregated

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])