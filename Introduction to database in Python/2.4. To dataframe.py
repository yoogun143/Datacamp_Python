import os
os.chdir('E:\Datacamp\Python\Introduction to database in Python')
from sqlalchemy import create_engine, select
engine = create_engine("sqlite:///census.sqlite")
connection = engine.connect()
# Import packages
from sqlalchemy import MetaData, Table

# Creaate metadata
metadata = MetaData()

# Reflect census table from the engine: census
census = Table("census", metadata, autoload=True, autoload_with=engine)

stmt = select([census])
results = connection.execute(stmt).fetchall()


#### RESULTSPROXY TO DATAFRAME, GRAPH
# import pandas
import pandas as pd
import matplotlib as plt

# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set column names
df.columns = results[0].keys()

# Print the Dataframe
print(df)

# Plot the DataFrame
df[10:20].plot.bar()
plt.show()
