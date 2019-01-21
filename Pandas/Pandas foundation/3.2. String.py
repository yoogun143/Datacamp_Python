import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Pandas foundation')
sales = pd.read_csv('sales-feb-2015.csv',
                    parse_dates=['Date'])

# String methods
sales['Company'].str.upper()

# Substring matching
sales['Product'].str.contains('ware').sum()
