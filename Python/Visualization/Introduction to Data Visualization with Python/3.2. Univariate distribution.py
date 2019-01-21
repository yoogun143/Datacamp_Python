import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Introduction to Data Visualization with Python')
import matplotlib.pyplot as plt
auto = pd.read_csv('auto-mpg.csv')
import seaborn as sns


#### STRIP PLOT
# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2,1,1)
sns.stripplot(x="cyl", y="hp", data=auto)

# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x = "cyl", y = "hp", data = auto, jitter = True, size = 3)

# Display the plot
plt.show()


#### SWARM PLOT
# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'  
plt.subplot(2,1,1)
sns.swarmplot(x = "cyl", y = "hp", data = auto)

# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2,1,2)
sns.swarmplot(x = "hp", y = "cyl", data = auto, orient = "h", hue = "origin")

# Display the plot
plt.show()


#### VIOLIN PLOT
# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2,1,1)
sns.violinplot(x="cyl", y="hp", data =auto)

# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2,1,2)
sns.violinplot(x = "cyl", y = "hp", inner = None, color = "lightgray", data = auto)

# Overlay a strip plot on the violin plot
sns.stripplot(x = "cyl", y = "hp", jitter = True, size = 1.5, data = auto)

# Display the plot
plt.show()