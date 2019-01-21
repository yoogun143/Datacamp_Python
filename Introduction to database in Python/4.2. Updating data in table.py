import os
os.chdir('E:\Datacamp\Python\Introduction to database in Python')
from sqlalchemy import create_engine, select, insert
engine = create_engine("sqlite:///census.sqlite")
connection = engine.connect()
# Import packages
from sqlalchemy import MetaData, Table

# Creaate metadata
metadata = MetaData()

# Reflect state_fact table from the engine: census
state_fact = Table("state_fact", metadata, autoload=True, autoload_with=engine)


#### UPDATE MULTIPLE RECORDS
# Build a select statement: select_stmt
select_stmt = select([state_fact]).where(state_fact.columns.census_region_name == "West")

# Print the results of executing the select_stmt
print(connection.execute(select_stmt).fetchall())

# Build a statement to update the notes to 'The Wild West': stmt
from sqlalchemy import update
stmt = update(state_fact).values(notes= "The Wild West")

# Append a where clause to match the West census region records
stmt = stmt.where(state_fact.columns.census_region_name == "West")

# Execute the statement: results
results = connection.execute(stmt)

# Print rowcount
print(results.rowcount)

# Execute the select_stmt again to view the changes
print(connection.execute(select_stmt).fetchall())


#### CORRELATED UPDATE
# Create new table and insert into database
# Import Table, Column, String, Integer, Float, Boolean from sqlalchemy
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with state_name and fips_code: flat_census
flat_census = Table('flat_census', metadata,
             Column('state_name', String(256)),
             Column('fips_code', Integer())
)

# Use the metadata to create the table
metadata.create_all(engine)
values_list = [(None, '17'),
 (None, '34'),
 (None, '38'),
 (None, '41'),
 (None, '11'),
 (None, '55'),
 (None, '4'),
 (None, '5'),
 (None, '8'),
 (None, '15'),
 (None, '20'),
 (None, '22'),
 (None, '30'),
 (None, '31'),
 (None, '40'),
 (None, '16'),
 (None, '25'),
 (None, '26'),
 (None, '29'),
 (None, '37'),
 (None, '39'),
 (None, '44'),
 (None, '45'),
 (None, '56'),
 (None, '18'),
 (None, '42'),
 (None, '46'),
 (None, '47'),
 (None, '50'),
 (None, '2'),
 (None, '10'),
 (None, '21'),
 (None, '28'),
 (None, '51'),
 (None, '12'),
 (None, '24'),
 (None, '32'),
 (None, '53'),
 (None, '6'),
 (None, '9'),
 (None, '13'),
 (None, '19'),
 (None, '23'),
 (None, '33'),
 (None, '35'),
 (None, '48'),
 (None, '1'),
 (None, '27'),
 (None, '36'),
 (None, '49'),
 (None, '54')]

# Transform value list to dataframe
import pandas as pd
dataframe = pd.DataFrame(values_list)
dataframe.columns = ['state_name', 'fips_code']
dataframe

# Transform dataframe to a list of dictionary
help(dataframe.to_dict)
flat_census_values = dataframe.to_dict('records')

# Add flat_census to database
flat_census = Table("flat_census", metadata, autoload=True, autoload_with=engine)
stmt = insert(flat_census)
connection.execute(stmt, flat_census_values)


# Build a statement to select name from state_fact: stmt
fips_stmt = select([state_fact.columns.name])

# Append a where clause to Match the fips_state to flat_census fips_code
fips_stmt = fips_stmt.where(
    state_fact.columns.fips_state == flat_census.columns.fips_code)

# Build an update statement to set the name to fips_stmt: update_stmt
update_stmt = update(flat_census).values(state_name=fips_stmt)

# Execute update_stmt: results
results = connection.execute(update_stmt)

# Print rowcount
print(results.rowcount)

# Updated table
connection.execute(select([flat_census])).fetchall()


#### DELETING ALL RECORDS FROM A TABLE
# Import delete, select
from sqlalchemy import delete, select

# Build a statement to empty the census table: stmt
stmt = delete(flat_census)

# Execute the statement: results
results = connection.execute(stmt)

# Print affected rowcount
print(results.rowcount)

# Build a statement to select all records from the census table
stmt = select([flat_census])

# Print the results of executing the statement to verify there are no rows
print(connection.execute(stmt).fetchall())


#### DELETER SPECIFIC RECORDS
census = Table("census", metadata, autoload=True, autoload_with=engine)
from sqlalchemy import func, and_

# Build a statement to count records using the sex column for Men ('M') age 36: stmt
stmt = select([func.count(census.columns.sex)]).where(
    and_(census.columns.sex == 'M',
         census.columns.age == 36)
)

# Execute the select statement and use the scalar() fetch method to save the record count
to_delete = connection.execute(stmt).scalar()

# Build a statement to delete records from the census table: stmt_del
stmt_del = delete(census)

# Append a where clause to target Men ('M') age 36
stmt_del = stmt_del.where(
    and_(census.columns.sex == "M",
         census.columns.age == 36)
)

# Execute the statement: results
results = connection.execute(stmt_del)

# Print affected rowcount and to_delete record count, make sure they match
print(results.rowcount, to_delete)


#### DELETE A TABLE COMPLETELY
# Drop the state_fact table
state_fact.drop(engine)

# Check to see if state_fact exists
print(state_fact.exists(engine))
