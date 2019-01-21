#### ACCESS DICTIONARY
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe["norway"])


#### DICTIONARY MANIPULATION
# Add italy to europe
europe["italy"] = "rome"

# Check italy in europe
print("italy" in europe)

# Update capital of germany
europe["germany"] = "bonn"

# Remove germany
del(europe["germany"])

# Print europe
print(europe)


# DICTIONARY OF DICTIONARIES
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe["france"]["capital"])

# Create sub-dictionary data
data = {"capital": "rome", "population": 59.83}

# Add data to europe under key 'italy'
europe["italy"] = data

# Print europe
print(europe)


#### USE DICTIONARY TO COUNT: Toolbox (part1)