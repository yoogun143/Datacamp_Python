import numpy as np
import pandas as pd
A = np.arange(8).reshape(2,4) + 0.1
A
B = np.arange(6).reshape(2,3) + 0.2
B
C = np.arange(12).reshape(3,4) + 0.3
C

# Stack arrays horizontally
np.hstack([B, A])
np.concatenate([B, A], axis = 1)

# Stack array vertically
np.vstack([A, C])
np.concatenate([A, C], axis = 0)

# Incompatible array dimensions
np.concatenate([A, B], axis = 0) # Incompatible columns
np.concatenate([A, C], axis = 1) # incompatible rows

# Population & unemployment data
population = pd.read_csv('pop.txt', index_col = 0)
unemployment = pd.read_csv('unemp.txt', index_col = 0)
population
unemployment

# Converting to arrays
population_array = np.array(population)
population_array # Index info is lost
unemployment_array = np.array(unemployment)
print(unemployment_array)

# Manipulating data as arrays
np.concatenate([population_array, unemployment_array], axis = 1)
