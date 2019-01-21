import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Introduction to Data Visualization with Python')
auto = pd.read_csv('auto-mpg.csv')
import numpy as np
mpg = np.array(auto['mpg'])
hp = np.array(auto['hp'])
import matplotlib.pyplot as plt

#### RECTANGULAR BINNING
# Generate a 2-D histogram
plt.hist2d(hp, mpg, bins = (20, 20), range=((40, 235), (8, 48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()


#### HEXAGONAL BINNING
# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp, mpg, gridsize=(15, 12), extent=(40, 235, 8, 48))

           
# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()
