import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Visualization\Bokeh - Interactive Visualization')
file = pd.read_csv('literacy_birth_rate.csv')

fertility_latinamerica = file[file.Continent == 'LAT']['fertility']
fertility_africa = file[file.Continent == 'AF']['fertility']
fertility_asia = file[file.Continent == 'ASI']['fertility']
fertility_europe = file[file.Continent == 'EUR']['fertility']

female_literacy_latinamerica = file[file.Continent == 'LAT']['female literacy']
female_literacy_africa = file[file.Continent == 'AF']['female literacy']
female_literacy_asia = file[file.Continent == 'ASI']['female literacy']
female_literacy_europe = file[file.Continent == 'EUR']['female literacy']

p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')