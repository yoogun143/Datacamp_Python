import os
os.chdir('E:\Datacamp\Python\\Network analysis\Part 1')
import networkx as nx
import pandas as pd
data = pd.read_csv('airlines.csv')
import numpy as np

T = nx.DiGraph()

nodes = [(1, {'category': 'I', 'occupation': 'scientist'}),
 (3, {'category': 'P', 'occupation': 'politician'}),
 (4, {'category': 'D', 'occupation': 'celebrity'}),
 (5, {'category': 'I', 'occupation': 'politician'}),
 (6, {'category': 'D', 'occupation': 'politician'}),
 (7, {'category': 'D', 'occupation': 'scientist'}),
 (8, {'category': 'I', 'occupation': 'celebrity'}),
 (9, {'category': 'D', 'occupation': 'celebrity'}),
 (10, {'category': 'I', 'occupation': 'celebrity'}),
 (11, {'category': 'I', 'occupation': 'celebrity'}),
 (12, {'category': 'I', 'occupation': 'scientist'}),
 (13, {'category': 'D', 'occupation': 'celebrity'}),
 (14, {'category': 'I', 'occupation': 'celebrity'}),
 (15, {'category': 'D', 'occupation': 'politician'}),
 (16, {'category': 'D', 'occupation': 'celebrity'}),
 (17, {'category': 'P', 'occupation': 'politician'}),
 (18, {'category': 'P', 'occupation': 'scientist'}),
 (19, {'category': 'P', 'occupation': 'scientist'}),
 (20, {'category': 'D', 'occupation': 'politician'}),
 (21, {'category': 'I', 'occupation': 'politician'}),
 (22, {'category': 'D', 'occupation': 'celebrity'}),
 (23, {'category': 'D', 'occupation': 'scientist'}),
 (24, {'category': 'D', 'occupation': 'politician'}),
 (25, {'category': 'D', 'occupation': 'celebrity'}),
 (26, {'category': 'D', 'occupation': 'politician'}),
 (27, {'category': 'P', 'occupation': 'politician'}),
 (28, {'category': 'I', 'occupation': 'celebrity'}),
 (29, {'category': 'P', 'occupation': 'celebrity'}),
 (30, {'category': 'I', 'occupation': 'scientist'}),
 (31, {'category': 'P', 'occupation': 'scientist'}),
 (32, {'category': 'D', 'occupation': 'politician'}),
 (33, {'category': 'P', 'occupation': 'scientist'}),
 (34, {'category': 'I', 'occupation': 'politician'}),
 (35, {'category': 'D', 'occupation': 'politician'}),
 (36, {'category': 'I', 'occupation': 'scientist'}),
 (37, {'category': 'I', 'occupation': 'politician'}),
 (38, {'category': 'I', 'occupation': 'scientist'}),
 (39, {'category': 'P', 'occupation': 'scientist'}),
 (40, {'category': 'I', 'occupation': 'celebrity'}),
 (41, {'category': 'P', 'occupation': 'politician'}),
 (42, {'category': 'D', 'occupation': 'politician'}),
 (43, {'category': 'I', 'occupation': 'celebrity'}),
 (44, {'category': 'I', 'occupation': 'celebrity'}),
 (45, {'category': 'D', 'occupation': 'scientist'}),
 (46, {'category': 'P', 'occupation': 'politician'}),
 (47, {'category': 'P', 'occupation': 'celebrity'}),
 (48, {'category': 'P', 'occupation': 'scientist'}),
 (49, {'category': 'P', 'occupation': 'politician'})]

T.add_nodes_from(nodes)

import datetime

edges = [(1, 3, {'date': datetime.date(2012, 11, 17)}),
 (1, 4, {'date': datetime.date(2007, 6, 19)}),
 (1, 5, {'date': datetime.date(2014, 3, 18)}),
 (1, 6, {'date': datetime.date(2007, 3, 18)}),
 (1, 7, {'date': datetime.date(2011, 12, 19)}),
 (1, 8, {'date': datetime.date(2013, 12, 7)}),
 (1, 9, {'date': datetime.date(2009, 11, 9)}),
 (1, 10, {'date': datetime.date(2008, 10, 7)}),
 (1, 11, {'date': datetime.date(2008, 8, 14)}),
 (1, 12, {'date': datetime.date(2011, 3, 22)}),
 (1, 13, {'date': datetime.date(2014, 8, 3)}),
 (1, 14, {'date': datetime.date(2007, 5, 19)}),
 (1, 15, {'date': datetime.date(2009, 12, 13)}),
 (1, 16, {'date': datetime.date(2011, 4, 7)}),
 (1, 17, {'date': datetime.date(2013, 8, 2)}),
 (1, 18, {'date': datetime.date(2014, 11, 17)}),
 (1, 19, {'date': datetime.date(2013, 5, 20)}),
 (1, 20, {'date': datetime.date(2010, 12, 15)}),
 (1, 21, {'date': datetime.date(2010, 11, 27)}),
 (1, 22, {'date': datetime.date(2013, 9, 5)}),
 (1, 23, {'date': datetime.date(2013, 3, 1)}),
 (1, 24, {'date': datetime.date(2007, 7, 8)}),
 (1, 25, {'date': datetime.date(2010, 5, 23)}),
 (1, 26, {'date': datetime.date(2007, 9, 14)}),
 (1, 27, {'date': datetime.date(2013, 1, 24)}),
 (1, 28, {'date': datetime.date(2013, 6, 21)}),
 (1, 29, {'date': datetime.date(2010, 6, 28)}),
 (1, 30, {'date': datetime.date(2011, 12, 2)}),
 (1, 31, {'date': datetime.date(2010, 7, 24)}),
 (1, 32, {'date': datetime.date(2010, 7, 4)}),
 (1, 33, {'date': datetime.date(2013, 9, 28)}),
 (1, 34, {'date': datetime.date(2007, 3, 17)}),
 (1, 35, {'date': datetime.date(2013, 11, 7)}),
 (1, 36, {'date': datetime.date(2012, 8, 13)}),
 (1, 37, {'date': datetime.date(2009, 2, 19)}),
 (1, 38, {'date': datetime.date(2007, 3, 17)}),
 (1, 39, {'date': datetime.date(2011, 11, 15)}),
 (1, 40, {'date': datetime.date(2011, 12, 26)}),
 (1, 41, {'date': datetime.date(2010, 2, 14)}),
 (1, 42, {'date': datetime.date(2014, 4, 16)}),
 (1, 43, {'date': datetime.date(2010, 2, 28)}),
 (1, 44, {'date': datetime.date(2007, 11, 2)}),
 (1, 45, {'date': datetime.date(2008, 5, 17)}),
 (1, 46, {'date': datetime.date(2013, 11, 18)}),
 (1, 47, {'date': datetime.date(2010, 11, 14)}),
 (1, 48, {'date': datetime.date(2007, 8, 19)}),
 (1, 49, {'date': datetime.date(2012, 5, 11)}),
 (16, 48, {'date': datetime.date(2007, 10, 27)}),
 (16, 18, {'date': datetime.date(2009, 11, 14)}),
 (16, 35, {'date': datetime.date(2009, 4, 19)}),
 (16, 36, {'date': datetime.date(2007, 7, 14)}),
 (18, 16, {'date': datetime.date(2012, 5, 7)}),
 (18, 24, {'date': datetime.date(2014, 5, 4)}),
 (18, 35, {'date': datetime.date(2012, 6, 16)}),
 (18, 36, {'date': datetime.date(2012, 4, 25)}),
 (19, 35, {'date': datetime.date(2012, 6, 25)}),
 (19, 36, {'date': datetime.date(2010, 10, 14)}),
 (19, 5, {'date': datetime.date(2013, 4, 18)}),
 (19, 8, {'date': datetime.date(2013, 10, 6)}),
 (19, 11, {'date': datetime.date(2009, 8, 2)}),
 (19, 13, {'date': datetime.date(2008, 9, 23)}),
 (19, 15, {'date': datetime.date(2011, 11, 26)}),
 (19, 48, {'date': datetime.date(2010, 1, 22)}),
 (19, 17, {'date': datetime.date(2012, 6, 23)}),
 (19, 20, {'date': datetime.date(2013, 11, 20)}),
 (19, 21, {'date': datetime.date(2008, 7, 6)}),
 (19, 24, {'date': datetime.date(2009, 4, 12)}),
 (19, 37, {'date': datetime.date(2011, 12, 28)}),
 (19, 30, {'date': datetime.date(2012, 1, 22)}),
 (19, 31, {'date': datetime.date(2009, 1, 26)}),
 (28, 1, {'date': datetime.date(2012, 1, 13)}),
 (28, 5, {'date': datetime.date(2010, 9, 26)}),
 (28, 7, {'date': datetime.date(2013, 11, 14)}),
 (28, 8, {'date': datetime.date(2010, 7, 22)}),
 (28, 11, {'date': datetime.date(2013, 3, 17)}),
 (28, 14, {'date': datetime.date(2008, 10, 18)}),
 (28, 15, {'date': datetime.date(2008, 12, 9)}),
 (28, 17, {'date': datetime.date(2012, 1, 14)}),
 (28, 20, {'date': datetime.date(2012, 6, 28)}),
 (28, 21, {'date': datetime.date(2011, 10, 5)}),
 (28, 24, {'date': datetime.date(2007, 5, 19)}),
 (28, 25, {'date': datetime.date(2013, 1, 24)}),
 (28, 27, {'date': datetime.date(2008, 6, 28)}),
 (28, 29, {'date': datetime.date(2008, 5, 16)}),
 (28, 30, {'date': datetime.date(2013, 5, 8)}),
 (28, 31, {'date': datetime.date(2007, 7, 23)}),
 (28, 35, {'date': datetime.date(2010, 8, 4)}),
 (28, 36, {'date': datetime.date(2011, 10, 18)}),
 (28, 37, {'date': datetime.date(2011, 6, 2)}),
 (28, 44, {'date': datetime.date(2009, 5, 23)}),
 (28, 48, {'date': datetime.date(2010, 10, 14)}),
 (28, 49, {'date': datetime.date(2013, 7, 17)}),
 (36, 24, {'date': datetime.date(2008, 5, 19)}),
 (36, 35, {'date': datetime.date(2008, 3, 19)}),
 (36, 5, {'date': datetime.date(2010, 8, 14)}),
 (36, 37, {'date': datetime.date(2012, 6, 19)}),
 (37, 24, {'date': datetime.date(2013, 8, 12)}),
 (37, 35, {'date': datetime.date(2013, 7, 6)}),
 (37, 36, {'date': datetime.date(2014, 10, 11)}),
 (39, 1, {'date': datetime.date(2012, 7, 1)}),
 (39, 35, {'date': datetime.date(2013, 11, 5)}),
 (39, 36, {'date': datetime.date(2009, 11, 6)}),
 (39, 38, {'date': datetime.date(2009, 4, 19)}),
 (39, 33, {'date': datetime.date(2008, 8, 12)}),
 (39, 40, {'date': datetime.date(2012, 8, 8)}),
 (39, 41, {'date': datetime.date(2009, 8, 12)}),
 (39, 45, {'date': datetime.date(2012, 5, 27)}),
 (39, 24, {'date': datetime.date(2011, 9, 15)}),
 (42, 1, {'date': datetime.date(2013, 12, 19)}),
 (43, 48, {'date': datetime.date(2007, 12, 7)}),
 (43, 35, {'date': datetime.date(2008, 3, 4)}),
 (43, 36, {'date': datetime.date(2013, 9, 16)}),
 (43, 37, {'date': datetime.date(2009, 11, 22)}),
 (43, 24, {'date': datetime.date(2014, 9, 19)}),
 (43, 29, {'date': datetime.date(2008, 10, 20)}),
 (43, 47, {'date': datetime.date(2010, 12, 16)}),
 (45, 1, {'date': datetime.date(2013, 3, 15)}),
 (45, 39, {'date': datetime.date(2012, 4, 25)}),
 (45, 41, {'date': datetime.date(2009, 5, 10)})]

T.add_edges_from(edges)


# BASIC DRAWING
# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph to screen
nx.draw(T)
plt.show()


#### QUERIES ON GRAPH
# list of nodes from the graph T that have the 'occupation' label of 'scientist'
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']

# Use a list comprehension to get a list of edges from the graph T that were formed for at least 6 years, i.e., from before 1 Jan 2010.
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < datetime.date(2010, 1, 1)]
T.edges[45, 41]


#### WEIGHTS ON EDGES
# Set the weight of the edge
T.edges[19,5]
T.edges[19,5]['weight'] = 2
T.edges[19, 5]

# Iterate over all the edges involving 19 to be equal 1.1 (with metadata)
for u, v, d in T.edges(data=True):

    # Check if node 19 is involved
    if 19 in [u,v]:
    
        # Set the weight to 1.1
        T.edges[u, v]['weight'] = 1.1


#### CHECK WHETHER THERE ARE SELF-LOOPS IN GRAPH 
# self loop are edeges begin and end on the same node
# intuitve in trip networks, in which individuals begin at one location and end in another.
T.selfloop_edges()
T.number_of_selfloops()


#### VISULIZE WITH MATRIX PLOT
# Import nxviz
import nxviz as nv

# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)

# Draw m to the screen
m.draw()

# Display the plot
plt.show()

# Convert T to a matrix format: A
# each node is one column and one row, and an edge between the two nodes is indicated by the value 1.
A = nx.to_numpy_matrix(T)

# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())

# Check that the `category` metadata field is lost from each node
# only the weight metadata is preserved; all other metadata is lost
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()
    
    
#### VISUALIZE WITH CIRCOS PLOT
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot

# Create the CircosPlot object: c
c = CircosPlot(T)

# Draw c to the screen
c.draw()

# Display the plot
plt.show()


#### VISUALIZE WITH ARC PLOT
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import ArcPlot

# Create the un-customized ArcPlot object: a
a = ArcPlot(T)

# Draw a to the screen
a.draw()

# Display the plot
plt.show()

# Create the customized ArcPlot object: a2
#  the nodes are ordered and colored by the 'category'
a2 = ArcPlot(T, node_order='category', node_color='category')

# Draw a2 to the screen
a2.draw()

# Display the plot
plt.show()


#### DEGREE CENTRALITY
# Definition: number of neighbors a node have / numbers of neighbors that node could help
[a for a in T.neighbors(1)]

# number of neighbors that a node has: degree
T.degree

# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)
deg_cent

# Compute the maximum degree centrality: max_dc
max_dc = max(list(deg_cent.values()))

# Return the node with maximum degree centrality
# Iterate over the degree centrality dictionary
nodes = set() # 

for k, v in deg_cent.items():
# Check if the current value has the maximum degree centrality
    if v == max_dc:      
    # Add the current node to the set of nodes
        nodes.add(k)

nodes

# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()


#### SHORTEST PATH (BREADTH FIRST SEARCH)
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]
    
    for node in queue:  
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])
        
        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))
            
            # Place the appropriate return statement
            return False

path_exists(T, 1, 5)


#### BETWEENNESS CENTRALITY
# betweeness centrality of 1 node = Number of shortest paths through that node / all possible shortest paths
# application: bridges between 2 group, 
# High betweeness centrality often equivalent to low degree centrality (chap 2 page 21)
# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)

# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)

# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))

# Display the plot
plt.show()

# Compute maximum betweenness centrality: max_bc
max_bc = max(list(bet_cen.values()))

# Find the user(s) that have collaborated the most: prolific_collaborators
prolific_collaborators = [n for n, dc in bet_cen.items() if dc == max_bc]

# Print the most prolific collaborator(s)
print(prolific_collaborators)

#### INDETIFY TRIANGLE RELATIONSHIP
# Convert to undirected
T = T.to_undirected()

# Number of triangles for each node
nx.triangles(T)

from itertools import combinations

# Write a function that identifies all nodes in a triangle relationship with a given node.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1,n2):
        
            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)
            
            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)
            
    return triangle_nodes
    
# Nodes involved in triangle with node 1
nodes_in_triangle(T, 1)


#### OPEN TRIANGLE
# friend recommendation system: "A" knows "B" and "A" knows "C", then it's probable that "B" also knows "C".
### Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False
    
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
    
        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
        
            in_open_triangle = True
            
            break
            
    return in_open_triangle


### Compute the number of open triangles in T
num_open_triangles = 0

# Iterate over all the nodes in T
for n in T.nodes():

    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):
    
        # Increment num_open_triangles
        num_open_triangles += 1
        
print(num_open_triangles)


### Recommend people to connect
from collections import defaultdict

# Initialize the defaultdict: recommended
recommended = defaultdict(int)

# Iterate over all the nodes in G
for n, d in T.nodes(data=True):

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(T.neighbors(n), 2):
    
        # Check whether n1 and n2 do not have an edge
        if not T.has_edge(n1, n2):
        
            # Increment recommended
            recommended[((n1), (n2))] += 1

# Identify the top 10 pairs of users
all_counts = sorted(recommended.values())
top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
print(top10_pairs)


#### MAXIMAL CLIQUE
# cliques that cannot be extended by adding an adjacent edge
# => a community with each one knows others in that community
# Maximal clique
maximal_clique = sorted(nx.find_cliques(T), key=lambda x: len(x))
maximal_clique

# Identify the largest maximal clique: largest_max_clique
largest_max_clique = set(maximal_clique[-1])

# Create a subgraph from the largest_max_clique: T_lmc
T_lmc = T.subgraph(largest_max_clique)

# Copy to a new graph
T_lmc = T_lmc.copy()

# Go out 1 degree of separation
for node in T_lmc.nodes():
    T_lmc.add_nodes_from(T.neighbors(node))
    T_lmc.add_edges_from(zip([node]*len(list(T.neighbors(node))), T.neighbors(node)))

# Record each node's degree centrality score
for n in T_lmc.nodes():
    T_lmc.node[n]['degree centrality'] = nx.degree_centrality(T_lmc)[n]
        
# Create the ArcPlot object: a
a = ArcPlot(T_lmc, node_order = 'degree centrality')

# Draw the ArcPlot to the screen
a.draw()
plt.show()


#### SUBGRAPH
# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']

# Create the set of nodes: nodeset
# set can be used with union or intersect,...
# set can remove duplicate in a list automatically
# https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists
nodeset = set(nodes)

# Iterate over nodes
for n in nodes:

    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)
    
    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs) # can also use with 'append' method in list

# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()