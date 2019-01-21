import matplotlib.pyplot as plt
import os
os.chdir('E:\Datacamp\Python\Supervised learning with sklearn')


#### EDA
# deal with Region, which we dropped in previous exercises since we did not have tools to deal with it
# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


#### CREATE DUMMY VARIABLES
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
# Drop_first will drop unneeded dummy variable (in this case 'Region_America')
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)


#### REGRESSION WITH CATEGORICAL FEATURES
y = df_region['life'].values.reshape(-1,1)
X = df_region.drop('life', axis = 1).values.reshape(-1,13)

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv = 5)

# Print the cross-validated scores
print(ridge_cv)
