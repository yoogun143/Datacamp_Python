import os
os.chdir('E:\Datacamp\Python\Pandas\Merging Dataframe')
import pandas as pd


#### MULTIINDEX ON ROW
medal_types = ['bronze', 'silver', 'gold']
medals_list = []

for medal in medal_types:

    file_name = "%s_top5.csv" % medal

    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name, index_col='Country')
    
    # Append medal_df to medals
    medals_list.append(medal_df)

# If not specify keys => mess
medals = pd.concat(medals_list)
medals

# Concatenate medals: medals
medals = pd.concat(medals_list, keys=['bronze', 'silver', 'gold'])

# Print medals
print(medals)


#### SLICE MULTINDEXED DATAFRAMES
# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level=0)

# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])

# Print data about silver medals
print(medals_sorted.loc['silver'])

# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice

# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:, "United Kingdom"], : ])


#### MULTIINDEX ON COLUMNS
# Concatenate medals: medals
medals = pd.concat(medals_list, keys=['bronze', 'silver', 'gold'], axis = 1)
medals

# Index
medals['bronze']


#### CONCAT FROM DICT
bronze = pd.read_csv('bronze_top5.csv', index_col = 'Country')
silver = pd.read_csv('silver_top5.csv', index_col = 'Country')
gold = pd.read_csv('gold_top5.csv', index_col = 'Country')
medal_dict = {'bronze': bronze, 'silver': silver, 'gold': gold}
medals = pd.concat(medal_dict)
medals
