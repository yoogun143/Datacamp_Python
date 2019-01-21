# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

import os
os.chdir('E:\Datacamp\Python\Visualization\Bokeh - Interactive Visualization')
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv("sprint.csv")

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

#### SELECTION AND NON-SELCTION GLYPHS
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label = 'Year', y_axis_label = 'Time', tools = 'box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle(x = 'Year', y = 'Time', source = source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


#### HOVER GLYPHS
# import the HoverTool
from bokeh.models import HoverTool

# Add a circle glyphs to figure p
p.circle(x = 'Year', y = 'Time', source = source, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p 
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)


#### COLORMAPPING
#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Read in the CSV file: df
df = pd.read_csv("auto-mpg.csv")

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)
