import pandas as pd
northeast = pd.Series(['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'])
south = pd.Series(['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'])
midwest = pd.Series(['IL', 'IN', 'MN', 'MO', 'NE', 'ND', 'SD', 'IA', 'KS', 'MI', 'OH', 'WI'])
west = pd.Series(['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR','WA'])

#### APPEND()
east = northeast.append(south)
print(east)

# The appended Index
print(east.index)
print(east.loc[3])

# Using .reset_index()
new_east = northeast.append(south).reset_index(drop=True)
print(new_east.head(11))
print(new_east.index)


#### CONCAT()
east = pd.concat([northeast, south])
print(east.head(11))
print(east.index)

# Using ignore_index
new_east = pd.concat([northeast, south], ignore_index = True)
print(new_east.head(11))
print(new_east.index)
