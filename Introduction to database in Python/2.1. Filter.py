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


#### FILTER DATA: SIMPLE
# Create a select query: stmt
stmt = select([census])

# Add a where clause to filter the results to only those for New York
stmt = stmt.where(census.columns.state == "New York")

# Execute the query to retrieve all the data returned: results
results = connection.execute(stmt).fetchall()

# Loop over the results and print the age, sex, and pop2008
for result in results:
    print(result.age, result.sex, result.pop2008)


#### FILTER DATA: EXPRESSION
# Create a query for the census table: stmt
stmt = select([census])

# Define states
states = ['New York', 'California', 'Texas']

# Append a where clause to match all the states in_ the list states
stmt = stmt.where(census.columns.state.in_(states))

# Loop over the ResultProxy and print the state and its population in 2000
for cen in connection.execute(stmt):
    print(cen.state, cen.pop2000)


#### FILTER DATA: ADVANCED
# Import and_
from sqlalchemy import and_, or_

# Build a query for the census table: stmt
stmt = select([census])

# Append a where clause to select people in NewYork who are 21 or 37 years old
stmt = stmt.where(
  and_(census.columns.state == 'New York',
       or_(census.columns.age == 21,
          census.columns.age == 37
         )
      )
  )

# Loop over the ResultProxy printing the age and sex
for result in connection.execute(stmt):
    print(result.state, result.age)