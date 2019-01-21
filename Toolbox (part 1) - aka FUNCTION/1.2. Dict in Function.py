import os
os.chdir('E:\Datacamp\Python\Toolbox (part 1) - aka FUNCTION')
import pandas as pd
tweets_df = pd.read_csv('tweets.csv')

# Define count_entries()
def count_entries(df, col_name = "lang"):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result1
result1 = count_entries(tweets_df)

# Call count_entries(): result2
result2 = count_entries(tweets_df, "source")

# Print result1 and result2
print(result1)
print(result2)
