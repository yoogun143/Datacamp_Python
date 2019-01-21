import os
import pandas as pd
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')
tips = pd.read_csv('tips.csv')


#### CHOOSE APPROPRIATE DATA TYPE
# Examine the datatype
tips.info()

# Convert the sex column to type 'category'
tips.sex = tips.sex.astype("category")

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype("category")

# Convert 'total_bill' to a numeric dtype
tips['total_bill'] = pd.to_numeric(tips["total_bill"], errors="coerce")

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips["tip"], errors="coerce")

# Print the info of tips
print(tips.info())
