import os
os.chdir('E:\Datacamp\Python\Pandas\Pandas foundation')
import pandas as pd

population = pd.read_csv('world_population.csv',
                         parse_dates=True, index_col = 'Year')

# Upsample population
population.resample('A').first()

# Interpolate missing data
population.resample('A').first().interpolate('linear')
