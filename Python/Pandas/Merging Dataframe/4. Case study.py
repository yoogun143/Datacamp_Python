import os
os.chdir('E:\Datacamp\Python\Pandas\Merging Dataframe')
import pandas as pd

#### PREPARE DATAFRAME
# Create file path: file_path
file_path = 'Summer Olympic medalists 1896 to 2008 - EDITIONS.tsv'

# Load DataFrame from file_path: editions
editions = pd.read_csv(file_path, sep="\t")

# Extract the relevant columns: editions
editions = editions[["Edition", "Grand Total", "City", "Country"]]

# Print editions DataFrame
print(editions)


#### LOAD IOC CODES DATAFRAMES
# Import pandas
import pandas as pd

# Create the file path: file_path
file_path = 'Summer Olympic medalists 1896 to 2008 - IOC COUNTRY CODES.csv'

# Load DataFrame from file_path: ioc_codes
ioc_codes = pd.read_csv(file_path)

# Extract the relevant columns: ioc_codes
ioc_codes = ioc_codes[["Country", "NOC"]]

# Print first and last 5 rows of ioc_codes
print(ioc_codes.head())
print(ioc_codes.tail())


#### BUILD MEDALS DATAFRAME (Do not run, do not have data)
# Create empty dictionary: medals_dict
medals_dict = {}

for year in editions['Edition']:

    # Create the file path: file_path
    file_path = 'summer_{:d}.csv'.format(year)
    
    # Load file_path into a DataFrame: medals_dict[year]
    medals_dict[year] = pd.read_csv(file_path)
    
    # Extract relevant columns: medals_dict[year]
    medals_dict[year] = medals_dict[year][["Athlete", "NOC", "Medal"]]
    
    # Assign year to column 'Edition' of medals_dict
    medals_dict[year]['Edition'] = year
    
# Concatenate medals_dict: medals
medals = pd.concat(medals_dict, ignore_index=True)


#### REAL MEDALS 
medals = pd.read_csv('Summer Olympic medalists 1896 to 2008 - ALL MEDALISTS.tsv', 
                     sep = '\t', header = 4)
medals = medals[["Athlete", "NOC", "Medal", "Edition"]]


#### COUNT MEDALS BY COUNTRY/EDITION
# Construct the pivot_table: medal_counts
medal_counts = medals.pivot_table(index = "Edition", values="Athlete", columns="NOC", aggfunc="count")

# Print the first & last 5 rows of medal_counts
print(medal_counts.head())
print(medal_counts.tail())


#### FRACTION OF MEDALS PER EDITION
# Set Index of editions: totals
totals = editions.set_index("Edition")

# Reassign totals['Grand Total']: totals
totals = totals["Grand Total"]

# Divide medal_counts by totals: fractions
fractions = medal_counts.divide(totals, axis = "rows")

# Print first & last 5 rows of fractions
print(fractions.head())
print(fractions.tail())


#### PERCENTAGE CHANGE IN FRACTION OF MEDALS WON
# Apply the expanding mean: mean_fractions = value of mean with all data available up to that point in time
mean_fractions = fractions.expanding().mean()

# Compute the percentage change: fractions_change
fractions_change = mean_fractions.pct_change() * 100

# Reset the index of fractions_change: fractions_change
fractions_change = fractions_change.reset_index()

# Print first & last 5 rows of fractions_change
print(fractions_change.head())
print(fractions_change.tail())


#### BUILD HOSTS DATAFRAME
# Left join editions and ioc_codes: hosts
hosts = pd.merge(editions, ioc_codes, how = "left")

# Extract relevant columns and set index: hosts
hosts = hosts[["Edition", "NOC"]].set_index("Edition")

# Fix missing 'NOC' values of hosts
print(hosts.loc[hosts.NOC.isnull()])
hosts.loc[1972, 'NOC'] = 'FRG'
hosts.loc[1980, 'NOC'] = 'URS'
hosts.loc[1988, 'NOC'] = 'KOR'

# Reset Index of hosts: hosts
hosts = hosts.reset_index()

# Print hosts
print(hosts)


#### RESHAPE FOR ANALYSIS
# Reshape fractions_change: reshaped
reshaped = pd.melt(fractions_change, id_vars="Edition", value_name="Change")

# Print reshaped.shape and fractions_change.shape
print(reshaped.shape, fractions_change.shape)

# Extract rows from reshaped where 'NOC' == 'CHN': chn
chn = reshaped[reshaped["NOC"] == "CHN"]

# Print last 5 rows of chn with .tail()
print(chn.tail())


#### SUMMARIZE FRACTIONAL CHANGE IN EXPANDING MEAN OF PERCENTAGE OF MEDALS WON FOR THE HOST COUNTRY IN EACH EDITION
# Merge reshaped and hosts: merged
merged = pd.merge(reshaped, hosts)

# Print first 5 rows of merged
print(merged.head())

# Set Index of merged and sort it: influence
influence = merged.set_index("Edition").sort_index()

# Print first 5 rows of influence
print(influence.head())


#### PLOT INFLUENCE OF HOST COUNTRY
# Import pyplot
import matplotlib.pyplot as plt

# Extract influence['Change']: change
change = influence["Change"]

# Make bar plot of change: ax
ax = change.plot(kind="bar")

# Customize the plot to improve readability
ax.set_ylabel("% Change of Host Country Medal Count")
ax.set_title("Is there a Host Country Advantage?")
ax.set_xticklabels(editions['City'])

# Display the plot
plt.show()