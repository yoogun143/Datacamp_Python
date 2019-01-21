import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Introduction to Data Visualization with Python')
import matplotlib.pyplot as plt
auto = pd.read_csv('auto-mpg.csv')
import seaborn as sns


#### JOINTPLOT
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.jointplot(x = "hp", y = "mpg", kind = "hex", data = auto)
help(sns.jointplot)

# Display the plot
plt.show()


#### PLOT DISTRIBUTIONS PAIRWISE
# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(auto, kind = "reg", hue = "origin")

# Display the plot
plt.show()


#### COVARIANCE MATRIX HEATMAP
# Visualize the covariance matrix using a heatmap
sns.heatmap(auto.corr())

# Display the heatmap
plt.show()
