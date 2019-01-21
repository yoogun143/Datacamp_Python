#### NUMPY
# Import numpy as np
import numpy as np

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create array using np.linspace: x
x = np.linspace(0, 5, 100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p = figure()
p.circle(x, y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)


#### DATAFRAME
# Import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Bokeh - Interactive Visualization')
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv("auto-mpg.csv")

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color = df['color'], size = 10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)


#### COLUMN DATA SOURCE = USED FOR BOKEH
df = pd.read_csv('sprint.csv')
# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)
source.data

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source = source, color = 'color', size = 8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)
