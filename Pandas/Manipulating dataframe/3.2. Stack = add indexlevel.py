import pandas as pd
import os
os.chdir('E:\Datacamp\Python\Pandas\Manipulating dataframe')
users = pd.read_csv('users.csv', index_col = ['city', 'weekday'])
users = users.sort_index()
users

# Unstack users by 'city': bycity
bycity = users.unstack(level="city")

# Print the bycity DataFrame
print(bycity)

# Stack bycity by 'city' and print it
print(bycity.stack(level="city")) # Not the same layout => need to swap level

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level="city")

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))
