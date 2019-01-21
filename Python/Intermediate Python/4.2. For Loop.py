#### LOOP OVER A LIST
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for area in areas:
    print(area)
    
# Change for loop to use enumerate()
for index, a in enumerate(areas) :
    print("room " + str(index) + ": " + str(a))
    
# Code the for loop
for index, area in enumerate(areas) :
    print("room " + str(index + 1) + ": " + str(area))
    

#### LOOP OVER LIST OF LISTS
# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for room in house:
    print("the", room[0], "is", str(room[1]), "sqm")
    
    
#### LOOP OVER DICTIONARY
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
          
# Iterate over europe
for key, value in europe.items() :
    print("the capital of " + key + " is " + value)


#### LOOP OVER NUMPY ARRAY
# Import numpy as np
import numpy as np
np_height = np.array([75, 53, 66, 23, 67, 24])
np_baseball = np.array([[75, 56], [65, 95], [24, 96], [21, 65]])

# For loop over np_height
for height in np_height:
    print(str(height), "inches")

# For loop over np_baseball
for stat in np.nditer(np_baseball) :
    print(stat)
    

#### LOOP OVER DATAFRAME
import os
os.chdir('E:\Datacamp\Python\Intermediate Python')

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows() :
    print(lab)
    print(row)
    
# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row["cars_per_cap"]))
    
    
#### ADD COLUMN TO DATAFRAME
# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows() :
    cars.loc[lab, "COUNTRY"] = str.upper(row["country"])

# Print cars
print(cars)

# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)
    
print(cars)
