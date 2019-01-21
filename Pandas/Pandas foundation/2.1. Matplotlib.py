import os
os.chdir('E:\Datacamp\Python\Pandas\Pandas foundation')
import pandas as pd
import matplotlib as plt
df = pd.read_csv('messy_stock_data.tsv.txt', header = 3, delimiter = " ", comment = '#')
df = df.transpose()
df.columns = df.iloc[0, :]
df = df.iloc[1:, :]
df = df.reset_index()

#### LINE PLOT
# Create a list of y-axis column names: y_columns
y_columns = ["APPLE", "IBM"]

# Generate a line plot
df.plot(x="index", y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()


#### SCATTER PLOT
df = pd.read_csv('auto-mpg.csv')
# Generate a scatter plot
df.plot(kind="scatter", x='hp', y='mpg', s=df['weight']/100)

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()


#### BOXPLOT
# Make a list of the column names to be plotted: cols
cols = ["weight", "mpg"]

# Generate the box plots
df[cols].plot(kind="box", subplots=True)

# Display the plot
plt.show()


#### HIST, PDF AND CDF
df = pd.read_csv('tips.csv')
# This formats the plots such that they appear on separate rows
fig, axes = plt.pyplot.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))

# Plot the CDF
df.fraction.plot(ax=axes[1], kind="hist", bins=30, normed=True, cumulative=True, range=(0,.3))
