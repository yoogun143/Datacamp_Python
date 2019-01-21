import os
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')

import pandas as pd
airquality = pd.read_csv('airquality.csv')
tb = pd.read_csv('tb.csv')
ebola = pd.read_csv('ebola.csv')

#### WIDE TO LONG DATA
# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(frame=airquality, 
                          id_vars=["Month", "Day"], 
                          var_name="measurement", 
                          value_name="reading")

# Print the head of airquality_melt
print(airquality_melt.head())


#### LONG TO WIDE DATA
# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=["Month", "Day"], 
                                               columns="measurement", 
                                               values="reading")

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the new index of airquality_pivot
print(airquality_pivot.index)

# Print the head of airquality_pivot
print(airquality_pivot.head())

# In case we do not select any particular variables, all of them are pivoted (visitors & signup)
users = pd.read_csv('users.csv', index_col = 0)

# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index="weekday", columns="city")

# Print the pivoted DataFrame
print(pivot)


# LONG TO WIDE DATA WHEN EXISTS DUPLICATE VALUES
# Use pivot table and define aggfunc = np.mean


#### OTHER FUNCTIONS IN PIVOT TABLE
# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index="weekday", aggfunc="count", margins = True)

# Print count_by_weekday
print(count_by_weekday1)


#### SPLITTING COLUMNS = SEPERATE IN R
# Melt tb: tb_melt
tb_melt = pd.melt(frame = tb, id_vars=["country", "year"])

# Create the 'gender' column
tb_melt['gender'] = tb_melt.variable.str[0]

# Create the 'age_group' column
tb_melt['age_group'] = tb_melt.variable.str[1:]

# Print the head of tb_melt
print(tb_melt.head())

# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=["Date", "Day"], 
                     var_name="type_country", 
                     value_name="counts")

# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split("_")

# Create the 'type' column
ebola_melt['type'] = ebola_melt.str_split.str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt.str_split.str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())
