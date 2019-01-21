# Empire State Building bet:
 # Roll a dice, if the dice = 1,2; go one step down
 # If dice = 3,4,5 => go one step up
 # Else, you throw the dice again. The number of eyes is the number of steps you go up.
 # You're a bit clumsy and you have a 0.1% chance of falling down. 
 # => What are the odds that you'll reach 60 steps high on the Empire State Building after 100 dice throws?
 
# Initialization
import numpy as np
np.random.seed(123)
random_walk = [0]


#### SINGLE WALK
for x in range(100) :
    step = random_walk[-1]
    dice = np.random.randint(1,7)

    if dice <= 2:
        step = max(0, step - 1) # make sure you cannot go below 0
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

# Show the plot
plt.show()


#### MULTIPLE WALK
# Initialize all_walks
all_walks = []

# Simulate random walk 250 times
for i in range(250):

    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
            
        # Implement clumsiness
        if np.random.rand() < 0.001 :
            step = 0
            
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

# Convert all_walks to Numpy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()
# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()

# Select last row from np_aw_t: ends
ends = np_aw_t[-1]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()

# Calculate the odds
np.mean(ends > 30)
