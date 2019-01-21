import os
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')

# Import pandas
import pandas as pd

# Read the file into a DataFrame: df
df = pd.read_csv('dob_job_application_filings_subset.csv')

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)

# Print the info of df
print(df.info())


#### STATISTICS FOR NUMERICAL DATA
# Calulate summary statistics of df
print(df.describe())


#### STATISTICS FOR NON-NUMERICAL DATA
# Print the value counts for 'Borough':
print(df['Borough'].value_counts(dropna=False))


#### VISUALIZING SINGLE VARIABLE
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()


#### VISUALIZING MULTIPLE VARIABLES
# Import necessary modules
import pandas as pd

# Create the boxplot
df.boxplot(column="Existing Height", by="Borough", rot=90)

# Display the plot
plt.show()

# Create scatter plot
df.plot(kind="scatter", x="Existing Height", y="Proposed Height", rot=70)
plt.show()

