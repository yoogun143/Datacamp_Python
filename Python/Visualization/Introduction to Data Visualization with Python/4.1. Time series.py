import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Introduction to Data Visualization with Python')
import matplotlib.pyplot as plt
stocks = pd.read_csv('stocks.csv', index_col = 0, parse_dates = True)
aapl = stocks['AAPL']
ibm = stocks['IBM']
csco = stocks['CSCO']
msft = stocks['MSFT']


#### MULTIPLE TIMESERIES ON COMMON AXES
# Plot the aapl time series in blue
plt.plot(aapl, color='blue', label='AAPL')

# Plot the ibm time series in green
plt.plot(ibm, color='green', label='IBM')

# Plot the csco time series in red
plt.plot(csco, color='red', label='CSCO')

# Plot the msft time series in magenta
plt.plot(msft, color = "magenta", label = "MSFT")

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation = 60)

# Display the plot
plt.show()


#### MULTIPLE TIME SERIES SLICES
# Plot the series in the top subplot in blue
plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001 to 2011')
plt.plot(aapl, color='blue')

# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the sliced series in the top subplot in red
plt.subplot(2, 1, 2)
plt.xticks(rotation = 45)
plt.title('AAPL: Nov. 2007 to Apr. 2008')
plt.plot(view, color = "red")

# Improve spacing and display the plot
plt.tight_layout()
plt.show()


#### INSET VIEW
# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the entire series 
plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')

# Specify the axes
plt.axes([0.25, 0.5, 0.35, 0.35])

# Plot the sliced series in red using the current axes
plt.plot(view, color = "red")
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()


#### MOVING AVERAGE
# Create 30-day moving average
mean_30 = aapl.resample('D').mean().rolling(window='30D').mean()
mean_250 = aapl.resample('D').mean().rolling(window='250D').mean()

# Plot the 30-day moving average in the top left subplot in green
plt.subplot(2, 1, 1)
plt.plot(mean_30, color = "green")
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

# Plot the 250-day moving average in the bottom right subplot in cyan
plt.subplot(2, 1, 2)
plt.plot(mean_250, "cyan")
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')

# Display the plot
plt.show()