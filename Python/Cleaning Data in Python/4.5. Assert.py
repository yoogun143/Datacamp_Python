import os
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')
import numpy as np
import pandas as pd
ebola = pd.read_csv('ebola.csv')
ebola = ebola.fillna(0)


# Assert that there are no missing values
assert pd.notnull(ebola).all().all()

# Assert that all values are >= 0
assert (ebola >= 0).all().all()
