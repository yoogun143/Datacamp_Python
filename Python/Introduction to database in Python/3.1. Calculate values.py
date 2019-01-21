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


#### CALCULATING A DIFFERENCE BETWEEN 2 COLUMNS
# Build query to return state names by population difference from 2008 to 2000: stmt
stmt = select([census.columns.state, (census.columns.pop2008-census.columns.pop2000).label("pop_change")])

# Append group by for the state: stmt
stmt = stmt.group_by(census.columns.state)

# Append order by for pop_change descendingly: stmt
from sqlalchemy import desc
stmt = stmt.order_by(desc("pop_change"))

# Return only 5 results: stmt
stmt = stmt.limit(5)

# Use connection to execute the statement and fetch all results
results = connection.execute(stmt).fetchall()

# Print the state and population change for each record
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))


#### OVERALL PERCENTAGE OF FEMALES
# import case, cast and Float from sqlalchemy
from sqlalchemy import case, cast, Float

# Build an expression to calculate female population in 2000
female_pop2000 = func.sum(
    case([
        (census.columns.sex == "F", census.columns.pop2000)
    ], else_=0))

# Cast an expression to calculate total population in 2000 to Float
total_pop2000 = cast(func.sum(census.columns.pop2000), Float)

# Build a query to calculate the percentage of females in 2000: stmt
stmt = select([female_pop2000 / total_pop2000* 100])

# Execute the query and store the scalar result: percent_female
percent_female = connection.execute(stmt).scalar()

# Print the percentage
print(percent_female)
