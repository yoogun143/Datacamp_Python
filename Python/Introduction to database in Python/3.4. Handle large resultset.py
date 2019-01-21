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

from sqlalchemy import select, func
stmt = select([census])
results_proxy = connection.execute(stmt)

state_count = {}
more_results = True

# Start a while loop checking for more results
while more_results:
    # Fetch the first 50 results from the ResultProxy: partial_results
    partial_results = results_proxy.fetchmany(50)

    # if empty list, set more_results to False
    if partial_results == []:
        more_results = False

    # Loop over the fetched records and increment the count for the state
    for row in partial_results:
        if row.state in state_count:
            state_count[row.state] += 1
        else:
            state_count[row.state] = 1

# Close the ResultProxy, and thus the connection
results_proxy.close()

# Print the count by state
print(state_count)