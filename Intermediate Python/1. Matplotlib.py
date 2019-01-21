import matplotlib.pyplot as plt
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]
values = [0,0.6,1.4,1.6,2.2,2.5,2.6,3.2,3.5,3.9,4.2,6]

# Scatter plot with log
plt.scatter(year, pop)
plt.xscale('log')
plt.show()

# Histogram
plt.hist(values, bins = 3)
plt.show()

# Add more data
year = [1800, 1850, 1900] + year
pop = [1.0, 1.262, 1.650] + pop

# Lineplot
plt.plot(year, pop)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0, 2, 4, 6, 8, 10],
['0', '2B', '4B', '6B', '8B', '10B'])
plt.show()