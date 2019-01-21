import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
titanic = pd.read_csv('titanic.csv')


# 1. OVERVIEW
#### GROUPBY MULTIPLE COLUMNS
# Group titanic by 'pclass'
by_class = titanic.groupby("pclass")

# Aggregate 'survived' column of by_class by count
count_by_class = by_class["survived"].count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(["embarked", "pclass"])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult["survived"].count()

# Print count_mult
print(count_mult)


#### GROUP BY ANOTHER SERIES
sales = pd.DataFrame({'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
                      'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
                      'bread': [139, 237, 326, 456],
                      'butter': [20, 45, 70, 98]})
customers = pd.Series(['Dave','Alice','Bob','Alice'])
sales.groupby(customers)['bread'].sum()


#### SPEED UP GROUPBY by CATEGORICAL DATA
# Find unique values
sales['weekday'].unique()

# Change to categorical data
sales['weekday'] = sales['weekday'].astype('category')

sales['weekday']


# 2. GROUPBY AND AGGREGATION
#### MULTIPLE AGGREGATIONS
# Aggregate max and sum for bread and butter groupby city
sales.groupby('city')[['bread','butter']].agg(['max','sum'])


#### CUSTOM AGGREGATION BY 1 FUNCTION
# Define function (I cannot use lambda here)
def data_range(series):
    return series.max() - series.min()

# Use function in groupby
sales.groupby('weekday')[['bread', 'butter']].agg(data_range)


#### CUSTOM AGGREGATION BY MULTIPLE FUNCTIONS
sales.groupby(customers)[['bread', 'butter']].agg({'bread':'sum', 'butter':data_range})


#### GROUPING ON FUNCTION OF INDEX
# Read file: sales1
sales1 = pd.read_csv("sales-feb-2015.csv", index_col="Date", parse_dates=True)

# Create a groupby object: by_day
by_day = sales1.groupby(sales1.index.strftime("%a"))

# Create sum: units_sum
units_sum = by_day["Units"].sum()

# Print units_sum
print(units_sum)


# 3. GROUPBY AND TRANSFORMATION
#### DETECTING OUTLIERS WITH Z-SCORE
auto = pd.read_csv('auto-mpg.csv')

# Define z-score
def zscore(series):
    return (series - series.mean()) / series.std()

# Define transformation and aggregation function
def zscore_with_year_and_name(group):
    df = pd.DataFrame({'mpg': zscore(group['mpg']),
                       'year': group['yr'],
                       'name': group['name']})
    return df

# Apply transformation and aggregation
auto.groupby('yr').apply(zscore_with_year_and_name).head()


#### FILLING MISSING DATA
# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(["sex", "pclass"])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = by_sex_class["age"].transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))


# 4. GROUPBY AND FILTER
#### GROUPBY AND FILTER WITH .APPLY()
# Define function
def c_deck_survival(gr):
    c_passengers = gr['cabin'].str.startswith('C').fillna(False)
    return gr.loc[c_passengers, 'survived'].mean()

# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby("sex")

# Call by_sex.apply with the function c_deck_survival and print the result
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)


#### GROUPBY AND FILTER WITH .FILTER()
sales = pd.read_csv('sales-feb-2015.csv', index_col = 'Date', parse_dates = True)

# Group sales by 'Company': by_company
by_company = sales.groupby("Company")

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company["Units"].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g["Units"].sum() > 35)
print(by_com_filt)


#### GROUPBY AND FILTER WITH .MAP()
# Create the Boolean Series: under10
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10).survived.mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10, "pclass"]).survived.mean()
print(survived_mean_2)


#### 5. BREAKDOWN GROUPBY OBJECT
# Create a groupby object
splitting = auto.groupby('yr')

# Type of groupby object
type(splitting)
type(splitting.groups)
print(splitting.groups.keys())

# Group by aggregation
splitting['mpg'].mean()

# Above aggregation equivalent to
for group_name, group in splitting:
    avg = group['mpg'].mean()
    print(group_name, avg)

# And we can filter in iteration
for group_name, group in splitting:
    avg = group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean()
    print(group_name, avg)
    
# This can be done in list comprehension
chevy_means = {year:group.loc[group['name'].str.contains('chevrolet'),'mpg'].mean() for year,group in splitting}
pd.Series(chevy_means)
