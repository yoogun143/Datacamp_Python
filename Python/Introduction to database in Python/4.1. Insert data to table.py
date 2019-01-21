import os
os.chdir('E:\Datacamp\Python\Introduction to database in Python')
from sqlalchemy import create_engine, select
engine = create_engine("sqlite:///:memory:") # In-memory database
connection = engine.connect()
# Import packages
from sqlalchemy import MetaData, Table

# Creaate metadata
metadata = MetaData()


##### CREATE TABLE
# Import Table, Column, String, Integer, Float, Boolean from sqlalchemy
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255), unique=True),
             Column('count', Integer(), default=1),
             Column('amount', Float()),
             Column('valid', Boolean(), default=False)
)

# Use the metadata to create the table
metadata.create_all(engine)

# Print the table details
print(repr(metadata.tables['data']))
data.constraints


#### INSERT 1 ROW
# Import insert and select from sqlalchemy
from sqlalchemy import insert, select

# Build an insert statement to insert a record into the data table: stmt
stmt = insert(data).values(name="Anna", count=1, amount=1000.00, valid=True)

# Execute the statement via the connection: results
results = connection.execute(stmt)

# Print result rowcount
print(results.rowcount)

# Build a select statement to validate the insert
stmt = select([data]).where(data.columns.name == "Anna")

# Print the result of executing the query.
print(connection.execute(stmt).first())


#### INSERT MULTIPLE ROWS
# Build a list of dictionaries: values_list
values_list = [
    {'name': "Thanh", 'count': 3, 'amount': 1000.00, 'valid': True},
    {"name": "Taylor", "count": 1, "amount": 750.00, "valid": False}
]

# Build an insert statement for the data table: stmt
stmt = insert(data)

# Execute stmt with the values_list: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)
connection.execute(select([data])).fetchall()


#### LOAD CSV INTO A TABLE
# Creaate metadata
metadata = MetaData()

# Define a new table
census = Table('census', metadata,
             Column('state', String(30)),
             Column('sex', String(1), default=1),
             Column('age', Integer()),
             Column('pop2000', Integer()),
             Column('pop2008', Integer())
)

# Use the metadata to create the table
metadata.create_all(engine)

# Create a insert statement for census: stmt
stmt = insert(census)

# Create an empty list and zeroed row count: values_list, total_rowcount
values_list = []
total_rowcount = 0

# Define csv_reader object
import csv
    
# Enumerate the rows of csv_reader
with open('census.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter = ",")
    for idx, row in enumerate(csv_reader):
        #create data and append to values_list
        data = {'state': row[0], 'sex': row[1], 'age': row[2], 'pop2000': row[3],
            'pop2008': row[4]}
        values_list.append(data)

        # Check to see if divisible by 51
        if idx % 51 == 0:
            results = connection.execute(stmt, values_list)
            total_rowcount += results.rowcount
            values_list = []

# Print total rowcount
print(total_rowcount)
connection.execute(select([census])).fetchall()
