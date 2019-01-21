import os
os.chdir('E:\Datacamp\Python\Cleaning Data in Python')
import numpy as np
import pandas as pd
import re
tips = pd.read_csv('tips.csv')
for lab, row in tips.iterrows():
    tips.loc[lab, 'total_dollar'] = '$' + str(tips.loc[lab, 'total_bill'])


#### RECODE MALE TO 1, FEMALE TO 0
# Define recode_sex()
def recode_sex(sex_value):

    # Return 1 if sex_value is 'Male'
    if sex_value == "Male":
        return 1
    
    # Return 0 if sex_value is 'Female'    
    elif sex_value == "Female":
        return 0
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['sex_recode'] = tips.sex.apply(recode_sex)

# Print the first five rows of tips
print(tips.head())


#### LAMBDA FUNCTION
# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())
