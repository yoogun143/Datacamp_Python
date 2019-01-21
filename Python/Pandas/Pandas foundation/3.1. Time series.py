# df3 = pd.read_csv(filename, index_col='Date', parse_dates=True)
import os
os.chdir('E:\Datacamp\Python\Pandas\Pandas foundation')
import pandas as pd

date_list0 = ['20100101 00:00',
 '20100101 01:00',
 '20100101 02:00',
 '20100101 03:00',
 '20100101 04:00']

date_list1 = ['20100101 00:00',
 '20100101 01:00',
 '20100101 02:00',
 '20100101 03:00',
 '20100101 04:00',
 '20100101 05:00',
 '20100101 06:00',
 '20100101 07:00',
 '20100101 08:00',
 '20100101 09:00']

#### CREATING TIME SERIES
temperature_list0 = [46.2, 44.6, 44.1, 43.8, 43.5]
temperature_list1 = [46.2, 44.6, 44.1, 43.8, 43.5, 43.0, 43.1, 42.3, 42.5, 45.9]

# Prepare a format string: time_format
time_format = "%Y-%m-%d %H:%M"

# Convert date_list into a datetime object: my_datetimes
my_datetimes0 = pd.to_datetime(date_list0, format=time_format)  
my_datetimes1 = pd.to_datetime(date_list1, format=time_format)

# Construct a pandas Series using temperature_list and my_datetimes: time_series
ts0 = pd.Series(temperature_list0, index=my_datetimes0)
ts1 = pd.Series(temperature_list1, index=my_datetimes1)


#### INDEXING TIME SERIES
# Extract the hour from 1am to 2am on '2010-10-01': tsa
tsa = ts0.loc['20100101 01:00:00']

# Extract '2010-01-01' from ts0: tsb
tsb = ts0.loc["2010-01-01"]

# Extract data from '2010-01-01' to '2010-12-31': tsc
tsc = ts0.loc["2010-01-01" : "2010-12-31"]


#### REINDEX AND COMBINE DATA
# Reindex without fill method: ts3
ts3 = ts0.reindex(ts1.index)
print(ts3)

# Reindex with fill method, using forward fill: ts4
ts4 = ts0.reindex(ts1.index, method="ffill")
print(ts4)

print(ts1 + ts0)
print(ts1 + ts3)
print(ts1 + ts4)


#### RESAMPLING
sales = pd.read_csv('sales-feb-2015.csv', parse_dates = True,
                    index_col = 'Date')
# Aggregating means
daily_mean = sales.resample('D').mean()
daily_mean

# Verifying
print(daily_mean.loc['2015-2-2'])
print(sales.loc['2015-2-2', 'Units'])
sales.loc['2015-2-2', 'Units'].mean()

# Method chaining
sales.resample('D').sum().max()

# Resampling strings
sales.resample('W').count()

# Multiplying frequencies
sales.loc[:,'Units'].resample('2W').sum()

# Upsampling and filling
sales.loc['2015-2-4': '2015-2-5', 'Units'].resample('4H').ffill()


#### ROLLING MEAN = MOVING AVERAGE
# Extract data daily unsmoothed
unsmoothed = sales.resample('D').mean().bfill()

# Applying a rolling mean with a 4 days window: smoothed
smoothed = unsmoothed.rolling(window = '4D').mean()

# Create a new DataFrame with columns smoothed and unsmoothed: MA
MA = pd.DataFrame({'smoothed':smoothed['Units'], 'unsmoothed':unsmoothed['Units']})

# Plot both smoothed and unsmoothed data using august.plot().
MA.plot()


#### DATETIME METHOD
sales['Date'].dt.hour


#### TIMEZONE
# Set timezone
central = sales['Date'].dt.tz_localize('US/Central')
central

# Convert timezone
central.dt.tz_convert('US/Eastern')


#### VISUALIZATION
import matplotlib as plt
sales.loc['2015-2-4': '2015-2-5','Units'].plot(style='k.-', kind = 'area',
                                               title = 'Sales ne')


#### CHANGE DATETIME INDEX
sales.index
# transform the index datetime values to abbreviated days of the week. 
sales.index.strftime("%a")
