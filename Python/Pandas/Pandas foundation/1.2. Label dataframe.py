import os
os.chdir('E:\Datacamp\Python\Pandas\Pandas foundation')
import pandas as pd
df = pd.DataFrame([['1980', 'Blondie', 'Call Me', '6'],
       ['1981', 'Chistorpher Cross', 'Arthurs Theme', '3'],
       ['1982', 'Joan Jett', 'I Love Rock and Roll', '7']])

# Build a list of labels: list_labels
list_labels = ["year", "artist", "song", "chart weeks"]

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels
