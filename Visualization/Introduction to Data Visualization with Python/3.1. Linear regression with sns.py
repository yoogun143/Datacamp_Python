import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Introduction to Data Visualization with Python')
import matplotlib.pyplot as plt
auto = pd.read_csv('auto-mpg.csv')


#### SIMPLE LINEAR REGRESSION
# Import plotting modules
import seaborn as sns

# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='hp', data=auto)

# Display the plot
plt.show()


#### RESIDUAL PLOT
# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')

# Display the plot
plt.show()


#### HIGHER-ORDER REGRESSION = POLYNOMIAL REGRESSION
# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto["mpg"], label='data', color='red', marker='o')

# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, color="blue", scatter=None, label='order 1')

# Plot in green a linear regression of order 2 between 'weight' and 'mpg' => polynomial regression
sns.regplot(x="weight", y = "mpg", data=auto, color="green", scatter=None, label="order 2", order = 2)

# Add a legend and display the plot
plt.legend(loc = "upper right")
plt.show()


#### GROUPING FACTOR IN SAME PLOT
# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x="weight", y="hp", hue="origin", data=auto, palette = "Set1")

# Display the plot
plt.show()


#### GROUPING FACTOR IN SUBPLOT
# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x = "weight", y = "hp", data = auto, row="origin")

# Display the plot
plt.show()
