import os
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')
import numpy as np
import pandas as pd
import re
tips = pd.read_csv('tips.csv')
airquality = pd.read_csv('airquality.csv')


#### DUPLICATE DATA
# Create the new DataFrame: tracks
tracks = tips[["sex", "smoker", "time"]]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks
print(tracks_no_duplicates.info())


#### FILL MISSING DATA
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality.Ozone.mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality.Ozone.fillna(oz_mean)

# Print the info of airquality
print(airquality.info())
