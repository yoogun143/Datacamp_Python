import os
import pandas as pd
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')

# Import the regular expression module
import re


#### STRING MATCH 
# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result = prog.match("1123-456-7890")
print(bool(result))


#### EXTRACT NUMERICAL VALUES
# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)
