import os
os.chdir('E:\Datacamp\Python\Pandas\Merging Dataframe')
import pandas as pd

#### NON-DUPLICATE INDEX
pop1 = pd.read_csv('pop1.txt', index_col = 0)
pop2 = pd.read_csv('pop2.txt', index_col = 0)

# Examining Data
pop1
pop2

# Appending population Dataframes
pop1.append(pop2)


#### DUPLICATE INDEX
population = pd.read_csv('pop.txt', index_col = 0)
unemployment = pd.read_csv('unemp.txt', index_col = 0)
population
unemployment

# Appending population & unemployment
population.append(unemployment) # => repeated index 2860

# Concat rows
pd.concat([population, unemployment], axis = 0) # => the same as above

# Concat columns = outer join
pd.concat([population, unemployment], axis = 1)# -> only 1 index 2860


#### OUTER JOIN
pd.concat([population, unemployment], axis = 1, join = 'outer')


#### INNER JOIN
pd.concat([population, unemployment], axis = 1, join = 'inner')
