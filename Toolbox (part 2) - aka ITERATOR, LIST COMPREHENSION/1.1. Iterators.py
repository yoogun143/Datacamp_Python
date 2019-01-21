#### ITERATE WITH FOR LOOP
# Iterate over a list
employees = ['Nick', 'Lore', 'Hugo']
for employee in employees:
    print(employee)
    
# Iterate over a string
for letter in 'Datacamp':
    print(letter)

# Iterate over a range
for i in range(4):
    print(i)


#### ITERATE OVER ITERABLES:
# New string: word
word = 'Da'

# Create an iterator for word
it = iter(word)

# Print each item for the iterator
next(it)
next(it)
next(it)


#### ITERATE OVER DICTIONARIES
pythonistas = {'hugo': 'bowne-anderson', 'francis':
'castro'}
for key, value in pythonistas.items():
    print(key, value)
    

#### ITERATE OVER FILE CONNECTIONS
import os
os.chdir('E:\Datacamp\Python\Toolbox (part 2)')
file = open('file.txt')
it = iter(file)
next(it)
next(it)


#### ITERATORS AS FUNCTION ARGUMENTS
# list() and sum() functions take iterators as arguments.
# Create a range object: values
values = range(10, 21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)


# USING ENUMERATE: ADD INDEX TO LIST
# enumerate() returns an enumerate object that produces a sequence of tuples, and each of the tuples is an index-value pair.
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pride']
aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
powers = ['telepathy',
 'thermokinesis',
 'teleportation',
 'magnetokinesis',
 'intangibility']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start = 1):
    print(index2, value2)
    
    
#### USING ZIP: ITERATE MULTIPLE LISTS
# takes any number of iterables and returns a zip object that is an iterator of tuples. 
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Can also create dict
dict(zip(mutants, aliases))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)


#### UNZIP
# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(list(result1) == mutants)
print(list(result2) == powers)
