import os
os.chdir('E:\Datacamp\Python\Introduction to database in Python')

# Import create_engine
from sqlalchemy import create_engine

# Create an engine that connects to the census.sqlite file: engine
engine = create_engine("sqlite:///census.sqlite")

# Create an engine with mysql
import pymysql
engine = create_engine('mysql+pymysql://' + 'student:datacamp' + '@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/' + 'census')
engine = create_engine('mysql+pymysql://' + 'root:1234' + '@localhost/' + 'test')

# Create an engine with PostgreSQL
import psycopg2
engine = create_engine('postgresql+psycopg2://' + 'student:datacamp' + '@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com' + ':5432/census')

# Print table names
print(engine.table_names())


# REFLECTION = READ DATABASE & BUILD METADATA
# Import packages
from sqlalchemy import MetaData, Table

# Creaate metadata
metadata = MetaData()

# Reflect census table from the engine: census
census = Table("census", metadata, autoload=True, autoload_with=engine)

# Print census table metadata
print(repr(census))

# Print the column names
print(census.columns.keys())

# Print full table metadata
print(repr(metadata.tables["census"]))


# SQL QUERY
# Build select statement for census table: stmt
stmt = "SELECT * FROM census"

# Alternative way of query with SQLAlchemy
from sqlalchemy import select
stmt = select([census])

# Create connection
connection = engine.connect()

# Execute the statement and fetch the results: results
results = connection.execute(stmt).fetchall()

# Print Results
print(results)


# HANDLE RESULTSET
# Get the first row of the results by using an index: first_row
first_row = results[0]

# Print the first row of the results
print(first_row)

# Print the first column of the first row by using an index
print(first_row[0])

# Print column names
print(first_row.keys())

# Print the 'state' column of the first row by using its name
print(first_row["state"])
