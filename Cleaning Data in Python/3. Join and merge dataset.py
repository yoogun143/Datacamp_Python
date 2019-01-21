import os
import pandas as pd
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')
uber1 = pd.read_csv('uber1.csv')
uber2 = pd.read_csv('uber2.csv')


#### COMBINING ROWS
# Concatenate uber1, uber2: row_concat
row_concat = pd.concat([uber1, uber2], ignore_index = True)

# Print the shape of row_concat
print(row_concat.shape)

# Print the head of row_concat
print(row_concat.head())


#### COMBINING COLUMNS
# Concatenate ebola_melt and status_country column-wise: ebola_tidy
uber_duplicate = pd.concat([uber1, uber2], axis = 1)

# Print the shape of ebola_tidy
print(uber_duplicate.shape)

# Print the head of ebola_tidy
print(uber_duplicate.head())


#### CONCATENATE THOUSANDS OF FILES AT ONCE
# Import necessary modules
import glob
import pandas as pd

# Write the pattern: pattern
pattern = 'uber*.csv'

# Save all file matches: csv_files
csv_files = glob.glob(pattern)

# Print the file names
print(csv_files)

# Load the second file into a DataFrame: csv2
csv2 = pd.read_csv(csv_files[1])

# Print the head of csv2
print(csv2.head())

# Create an empty list: frames
frames = []

#  Iterate over csv_files
for csv in csv_files:

    #  Read csv into a DataFrame: df
    df = pd.read_csv(csv)
    
    # Append df to frames
    frames.append(df)

# Concatenate frames into a single DataFrame: uber
uber = pd.concat(frames)

# Print the shape of uber
print(uber.shape)

# Print the head of uber
print(uber.head())


#### JOIN: SEE SLIDES