import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Bokeh - Interactive Visualization')
file = pd.read_csv('literacy_birth_rate.csv')
fertility_latinamerica = file[file.Continent == 'LAT']['fertility']
fertility_africa = file[file.Continent == 'AF']['fertility']
female_literacy_latinamerica = file[file.Continent == 'LAT']['female literacy']
female_literacy_africa = file[file.Continent == 'AF']['female literacy']

from datetime import datetime
aapl = pd.read_csv('aapl.csv')
date = list(aapl['date'])
date_datetime = []
for d in date:
    date_datetime.append(datetime.strptime(d, '%Y-%m-%d'))
price = list(aapl['close'])

#### SCATTER PLOT
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.x(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)


#### LINES & MARKERS
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date_datetime, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date_datetime, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)


#### PATCHES
xs = [ [1,1,2,2], [2,2,4], [2,2,3,3] ]
ys = [ [2,5,5,2], [3,5,5], [2,3,4,2] ]
p = figure()
p.patches(xs, ys, fill_color = ['red', 'blue', 'green'],
          line_color = 'white')
output_file('patches.html')
show(p)
