import os
os.chdir('E:\Datacamp\Python\Other\Statistical Thinking in Python')
versicolor_petal_length = [4.7, 4.5, 4.9, 4. , 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4. ,
       4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4. , 4.9, 4.7, 4.3, 4.4,
       4.8, 5. , 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1,
       4. , 4.4, 4.6, 4. , 3.3, 4.2, 4.2, 4.2, 4.3, 3. , 4.1]
setosa_petal_length = [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4,
       1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1. , 1.7, 1.9, 1.6,
       1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.5, 1.3,
       1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4]
virginica_petal_length = [6. , 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5,
       5. , 5.1, 5.3, 5.5, 6.7, 6.9, 5. , 5.7, 4.9, 6.7, 4.9, 5.7, 6. ,
       4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8,
       5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5. , 5.2, 5.4, 5.1]
versicolor_petal_width = [1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1. , 1.3, 1.4, 1. , 1.5, 1. ,
       1.4, 1.3, 1.4, 1.5, 1. , 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4,
       1.4, 1.7, 1.5, 1. , 1.1, 1. , 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3,
       1.3, 1.2, 1.4, 1.2, 1. , 1.3, 1.2, 1.3, 1.3, 1.1, 1.3]

#### HISTOGRAM
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Import numpy
import numpy as np

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Set default Seaborn style
sns.set()

# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length, bins=n_bins)

# Label axes
_ = plt.xlabel("petal length (cm)")
_ = plt.ylabel("count")

# Show histogram
plt.show()


#### BOX-AND-WHISKER PLOT
import seaborn as sns
import pandas as pd
df_all_states = pd.read_csv('2008_all_states.csv')

# Create box plot with Seaborn's default settings
_ = sns.boxplot(x='east_west', y='dem_share', data=df_all_states)

# Label the axes
_ = plt.xlabel("region")
_ = plt.ylabel("percent of vote for Obama")

# Show the plot
plt.show()


#### SCATTER PLOT
# Make a scatter plot
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker = ".", linestyle = "none")

# Set margins
_ = plt.margins(0.02)

# Label the axes
_ = plt.xlabel("Petal Length (cm)")
_ = plt.ylabel("Petal Width (cm)")

# Show the result
plt.show()


#### SWARM PLOT
import pandas as pd
df_swing = pd.read_csv('2008_swing_states.csv')

# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x='state', y='dem_share', data=df_swing)
# Label the axes
_ = plt.xlabel("state")
_ = plt.ylabel("percent of vote for Obama")

# Show the plot
plt.show()


#### ECDF
# Compute ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker = ".", linestyle = "none")
_ = plt.plot(x_vers, y_vers, marker = ".", linestyle = "none")
_ = plt.plot(x_virg, y_virg, marker = ".", linestyle = "none")

# Make nice margins
plt.margins(0.02)

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()
# =>Setosa is much shorter, also with less absolute variability in petal length than versicolor and virginica.