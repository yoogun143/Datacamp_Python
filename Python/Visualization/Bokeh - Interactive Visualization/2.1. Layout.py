import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Bokeh - Interactive Visualization')
file = pd.read_csv('literacy_birth_rate.csv')
# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(file)


#### ROW OF PLOTS (COLUMN)
# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle(x = 'fertility', y = 'female literacy', source = source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle(x = 'population', y = 'female literacy', source = source)

# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)


#### NEST ROWS AND COLUMNS (ERROR)
# Import column and row from bokeh.layouts
from bokeh.layouts import column, row

# Make a column layout that will be used as the second row: row2
row2 = column([p1, p2], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([None, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)


#### GRIDDED LAYOUT
# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [None, p1]

# Create a list containing plots p3 and p4: row2
row2 = [p2, None]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)


#### TABBED LAYOUT
# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel, Tabs

# Create a Panel with a title for each tab
first = Panel(child=p1, title='first')
second = Panel(child=p2, title='second')

# Put the Panels in a Tabs object
tabs = Tabs(tabs=[first, second])

# Specify the name of the output_file and show the result
output_file('tabbed.html')
show(tabs)


