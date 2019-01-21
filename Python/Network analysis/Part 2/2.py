import networkx as nx
nodes = [('u1570', {'bipartite': 'users'}),
 ('p45560', {'bipartite': 'projects'}),
 ('p81941', {'bipartite': 'projects'}),
 ('u16091', {'bipartite': 'users'}),
 ('u22413', {'bipartite': 'users'}),
 ('p2560', {'bipartite': 'projects'}),
 ('u9525', {'bipartite': 'users'}),
 ('u1798', {'bipartite': 'users'}),
 ('u4161', {'bipartite': 'users'}),
 ('u1955', {'bipartite': 'users'}),
 ('u10807', {'bipartite': 'users'}),
 ('p93377', {'bipartite': 'projects'}),
 ('u10968', {'bipartite': 'users'}),
 ('u47889', {'bipartite': 'users'}),
 ('p20663', {'bipartite': 'projects'}),
 ('p6322', {'bipartite': 'projects'}),
 ('u8326', {'bipartite': 'users'}),
 ('p33290', {'bipartite': 'projects'}),
 ('p64556', {'bipartite': 'projects'}),
 ('p79232', {'bipartite': 'projects'}),
 ('u5436', {'bipartite': 'users'}),
 ('p7303', {'bipartite': 'projects'}),
 ('u21', {'bipartite': 'users'}),
 ('p31240', {'bipartite': 'projects'}),
 ('p1979', {'bipartite': 'projects'}),
 ('p553', {'bipartite': 'projects'}),
 ('p397', {'bipartite': 'projects'}),
 ('p2627', {'bipartite': 'projects'}),
 ('u22296', {'bipartite': 'users'}),
 ('p97099', {'bipartite': 'projects'}),
 ('u28984', {'bipartite': 'users'}),
 ('p30186', {'bipartite': 'projects'}),
 ('p40810', {'bipartite': 'projects'}),
 ('u16961', {'bipartite': 'users'}),
 ('u4335', {'bipartite': 'users'}),
 ('u862', {'bipartite': 'users'}),
 ('p91402', {'bipartite': 'projects'}),
 ('p98819', {'bipartite': 'projects'}),
 ('u18069', {'bipartite': 'users'}),
 ('p1717', {'bipartite': 'projects'}),
 ('p2903', {'bipartite': 'projects'}),
 ('p25860', {'bipartite': 'projects'}),
 ('p3459', {'bipartite': 'projects'}),
 ('u4053', {'bipartite': 'users'}),
 ('u15014', {'bipartite': 'users'}),
 ('p3166', {'bipartite': 'projects'}),
 ('p878', {'bipartite': 'projects'}),
 ('u4412', {'bipartite': 'users'}),
 ('p11922', {'bipartite': 'projects'}),
 ('p60907', {'bipartite': 'projects'}),
 ('u14984', {'bipartite': 'users'}),
 ('u25116', {'bipartite': 'users'}),
 ('u929', {'bipartite': 'users'}),
 ('p2258', {'bipartite': 'projects'}),
 ('p920', {'bipartite': 'projects'}),
 ('u22913', {'bipartite': 'users'}),
 ('u35047', {'bipartite': 'users'}),
 ('u47508', {'bipartite': 'users'}),
 ('u18898', {'bipartite': 'users'}),
 ('u4966', {'bipartite': 'users'}),
 ('p20681', {'bipartite': 'projects'}),
 ('p80870', {'bipartite': 'projects'}),
 ('u10199', {'bipartite': 'users'}),
 ('p58090', {'bipartite': 'projects'}),
 ('p51216', {'bipartite': 'projects'}),
 ('u17865', {'bipartite': 'users'}),
 ('p34382', {'bipartite': 'projects'}),
 ('u2403', {'bipartite': 'users'}),
 ('p536', {'bipartite': 'projects'}),
 ('p1070', {'bipartite': 'projects'}),
 ('p37747', {'bipartite': 'projects'}),
 ('u17133', {'bipartite': 'users'}),
 ('u1975', {'bipartite': 'users'}),
 ('u5061', {'bipartite': 'users'}),
 ('u631', {'bipartite': 'users'}),
 ('p22173', {'bipartite': 'projects'}),
 ('p607', {'bipartite': 'projects'}),
 ('p19029', {'bipartite': 'projects'}),
 ('u6270', {'bipartite': 'users'}),
 ('u30668', {'bipartite': 'users'}),
 ('p5647', {'bipartite': 'projects'}),
 ('p93165', {'bipartite': 'projects'}),
 ('u322', {'bipartite': 'users'}),
 ('p53529', {'bipartite': 'projects'}),
 ('u4878', {'bipartite': 'users'}),
 ('u1047', {'bipartite': 'users'}),
 ('p70015', {'bipartite': 'projects'}),
 ('u39725', {'bipartite': 'users'}),
 ('u21567', {'bipartite': 'users'}),
 ('u1875', {'bipartite': 'users'}),
 ('u76', {'bipartite': 'users'}),
 ('p22200', {'bipartite': 'projects'}),
 ('p46614', {'bipartite': 'projects'}),
 ('p502', {'bipartite': 'projects'}),
 ('u45862', {'bipartite': 'users'}),
 ('p7059', {'bipartite': 'projects'}),
 ('u943', {'bipartite': 'users'}),
 ('p37180', {'bipartite': 'projects'}),
 ('p49227', {'bipartite': 'projects'}),
 ('p8', {'bipartite': 'projects'}),
 ('p29203', {'bipartite': 'projects'}),
 ('u4678', {'bipartite': 'users'}),
 ('u1832', {'bipartite': 'users'}),
 ('u1835', {'bipartite': 'users'}),
 ('p885', {'bipartite': 'projects'}),
 ('p9535', {'bipartite': 'projects'}),
 ('p3892', {'bipartite': 'projects'}),
 ('u640', {'bipartite': 'users'}),
 ('u12233', {'bipartite': 'users'}),
 ('u3528', {'bipartite': 'users'}),
 ('u363', {'bipartite': 'users'}),
 ('u25102', {'bipartite': 'users'}),
 ('u7161', {'bipartite': 'users'}),
 ('u21894', {'bipartite': 'users'}),
 ('u9531', {'bipartite': 'users'}),
 ('p53577', {'bipartite': 'projects'}),
 ('p6783', {'bipartite': 'projects'}),
 ('u5112', {'bipartite': 'users'}),
 ('u156', {'bipartite': 'users'}),
 ('u952', {'bipartite': 'users'}),
 ('u2800', {'bipartite': 'users'}),
 ('u3901', {'bipartite': 'users'}),
 ('u1336', {'bipartite': 'users'}),
 ('u5231', {'bipartite': 'users'}),
 ('p71650', {'bipartite': 'projects'}),
 ('p14187', {'bipartite': 'projects'}),
 ('p611', {'bipartite': 'projects'}),
 ('p16690', {'bipartite': 'projects'}),
 ('u53', {'bipartite': 'users'}),
 ('p750', {'bipartite': 'projects'}),
 ('u14318', {'bipartite': 'users'}),
 ('u627', {'bipartite': 'users'}),
 ('p81255', {'bipartite': 'projects'}),
 ('p2600', {'bipartite': 'projects'}),
 ('u211', {'bipartite': 'users'}),
 ('u44114', {'bipartite': 'users'}),
 ('u9284', {'bipartite': 'users'}),
 ('p11715', {'bipartite': 'projects'}),
 ('p91694', {'bipartite': 'projects'}),
 ('p776', {'bipartite': 'projects'}),
 ('u4560', {'bipartite': 'users'}),
 ('p162', {'bipartite': 'projects'}),
 ('p7618', {'bipartite': 'projects'}),
 ('u13790', {'bipartite': 'users'}),
 ('p360', {'bipartite': 'projects'}),
 ('p65', {'bipartite': 'projects'}),
 ('u14210', {'bipartite': 'users'}),
 ('u1266', {'bipartite': 'users'}),
 ('p4946', {'bipartite': 'projects'})]

edges = [('u1570', 'p2600', {}),
 ('u627', 'p162', {}),
 ('u627', 'p25860', {}),
 ('p81941', 'u44114', {}),
 ('u16091', 'p2903', {}),
 ('u22413', 'p93377', {}),
 ('p2560', 'u76', {}),
 ('u9525', 'p2600', {}),
 ('p70015', 'u39725', {}),
 ('u1798', 'p162', {}),
 ('u1798', 'p6783', {}),
 ('u1955', 'p65', {}),
 ('u1955', 'p502', {}),
 ('u1955', 'p71650', {}),
 ('u1955', 'p5647', {}),
 ('u10807', 'p11715', {}),
 ('u10968', 'p11922', {}),
 ('u47889', 'p3459', {}),
 ('p20663', 'u3528', {}),
 ('p6322', 'u5436', {}),
 ('u8326', 'p3166', {}),
 ('u3528', 'p7618', {}),
 ('u3528', 'p885', {}),
 ('u5436', 'p9535', {}),
 ('p7303', 'u6270', {}),
 ('u21', 'p51216', {}),
 ('p31240', 'u22913', {}),
 ('p1979', 'u7161', {}),
 ('p1979', 'u4878', {}),
 ('p1979', 'u1336', {}),
 ('p553', 'u4335', {}),
 ('p397', 'u28984', {}),
 ('p2627', 'u1835', {}),
 ('p97099', 'u25102', {}),
 ('p30186', 'u156', {}),
 ('p40810', 'u9284', {}),
 ('u16961', 'p2903', {}),
 ('u4335', 'p14187', {}),
 ('u862', 'p1070', {}),
 ('u862', 'p25860', {}),
 ('p91402', 'u21567', {}),
 ('u4412', 'p2903', {}),
 ('p1717', 'u18898', {}),
 ('p1717', 'u13790', {}),
 ('p1717', 'u1832', {}),
 ('p2903', 'u1975', {}),
 ('p2903', 'u5112', {}),
 ('p2903', 'u4053', {}),
 ('p2903', 'u2403', {}),
 ('p2903', 'u4161', {}),
 ('p2903', 'u640', {}),
 ('p2903', 'u4678', {}),
 ('p2903', 'u18069', {}),
 ('p25860', 'u929', {}),
 ('u14318', 'p16690', {}),
 ('p4946', 'u4966', {}),
 ('p3166', 'u943', {}),
 ('p3166', 'u5061', {}),
 ('p878', 'u1835', {}),
 ('p98819', 'u21894', {}),
 ('p60907', 'u10199', {}),
 ('u14984', 'p2600', {}),
 ('u25116', 'p37747', {}),
 ('p33290', 'u9531', {}),
 ('p920', 'u952', {}),
 ('u35047', 'p8', {}),
 ('u47508', 'p91694', {}),
 ('u4966', 'p1070', {}),
 ('u4966', 'p776', {}),
 ('u4966', 'p360', {}),
 ('u4966', 'p536', {}),
 ('p20681', 'u15014', {}),
 ('p58090', 'u14210', {}),
 ('u17865', 'p22173', {}),
 ('p34382', 'u1835', {}),
 ('u17133', 'p750', {}),
 ('u631', 'p611', {}),
 ('p607', 'u1266', {}),
 ('p19029', 'u1875', {}),
 ('u6270', 'p49227', {}),
 ('u6270', 'p81255', {}),
 ('p93165', 'u15014', {}),
 ('p53529', 'u5231', {}),
 ('u1047', 'p29203', {}),
 ('p22200', 'u45862', {}),
 ('p46614', 'u15014', {}),
 ('p37180', 'u1835', {}),
 ('u30668', 'p3892', {}),
 ('p29203', 'u211', {}),
 ('u1835', 'p7059', {}),
 ('u1835', 'p2258', {}),
 ('u12233', 'p64556', {}),
 ('u322', 'p611', {}),
 ('u322', 'p2258', {}),
 ('u322', 'p65', {}),
 ('u9531', 'p79232', {}),
 ('p53577', 'u22296', {}),
 ('u2800', 'p2600', {}),
 ('u3901', 'p80870', {}),
 ('u3901', 'p45560', {}),
 ('u363', 'p2600', {}),
 ('u53', 'p2600', {}),
 ('u4560', 'p2600', {})]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)


#### EDA
# number of nodes and edges
len(G.nodes())
len(G.edges())


#### PLOT WITH NXVIZ
from nxviz import CircosPlot
import matplotlib.pyplot as plt

# Add the degree centrality score of each node to their metadata dictionary
dcs = nx.degree_centrality(G)
for n in G.nodes():
    G.node[n]['centrality'] = dcs[n]
    
# Create the CircosPlot object: c
c = CircosPlot(G, node_color='bipartite', node_grouping='bipartite', node_order='centrality')

# Draw c to the screen
c.draw()

# Display the plot
plt.show()


#### BIPARTIE KEYWORD
# Biparties: a graph that is partitioned into 2 sets, nodes are only connected to nodes in other partitions
# in github data set, users only connect with projects, but no projects-projects or users-users
# Define get_nodes_from_partition()
def get_nodes_from_partition(G, partition):
    # Initialize an empty list for nodes to be returned
    nodes = []
    # Iterate over each node in the graph G
    for n in G.nodes():
        # Check that the node belongs to the particular partition
        if G.node[n]['bipartite'] == partition:
            # If so, append it to the list of nodes
            nodes.append(n)
    return nodes

# Import matplotlib
import matplotlib.pyplot as plt

# Get the 'users' nodes: user_nodes
user_nodes = get_nodes_from_partition(G, 'users')

# Compute the degree centralities: dcs
dcs = nx.degree_centrality(G)

# Get the degree centralities for user_nodes: user_dcs
user_dcs = [dcs[n] for n in user_nodes]

# Plot the degree distribution of users_dcs
plt.yscale('log')
plt.hist(user_dcs, bins=20)
plt.show()


#### RECOMMENDATION SYSTEM WITH BIPARTIE
# different from the previous recommendation system when we use maximal clique
# Shared nodes between 2 nodes in other partition
def shared_partition_nodes(G, node1, node2):
    # Check that the nodes belong to the same partition
    assert G.node[node1]['bipartite'] == G.node[node2]['bipartite']

    # Get neighbors of node 1: nbrs1
    nbrs1 = G.neighbors(node1)
    # Get neighbors of node 2: nbrs2
    nbrs2 = G.neighbors(node2)

    # Compute the overlap using set intersections
    overlap = set(nbrs1).intersection(nbrs2)
    return overlap

# Print the number of shared repositories between users 'u1570' and 'u16091'
print(len(shared_partition_nodes(G, 'u1570', 'u16091')))


# Function to compute metric of similarity between 2 users
# the number of projects shared between two users divided by the total number of nodes in the other partition.
def user_similarity(G, user1, user2, proj_nodes):
    # Check that the nodes belong to the 'users' partition
    assert G.node[user1]['bipartite'] == 'users'
    assert G.node[user2]['bipartite'] == 'users'

    # Get the set of nodes shared between the two users
    shared_nodes = shared_partition_nodes(G, user1, user2)

    # Return the fraction of nodes in the projects partition
    return len(shared_nodes) / len(proj_nodes)

# Compute the similarity score between users 'u1570' and 'u16091'
project_nodes = get_nodes_from_partition(G, 'projects')
similarity_score = user_similarity(G, 'u1570', 'u16091', project_nodes)

print(similarity_score)


# Function to finds the users most similar to another given user.
from collections import defaultdict

def most_similar_users(G, user, user_nodes, proj_nodes):
    # Data checks
    assert G.node[user]['bipartite'] == 'users'

    # Get other nodes from user partition
    user_nodes = set(user_nodes) 
    user_nodes.remove(user)

    # Create the dictionary: similarities
    similarities = defaultdict(list)
    for n in user_nodes:
        similarity = user_similarity(G, user, n, proj_nodes)
        similarities[similarity].append(n)

    # Compute maximum similarity score: max_similarity
    max_similarity = max(similarities.keys())

    # Return list of users that share maximal similarity
    return similarities[max_similarity]

user_nodes = get_nodes_from_partition(G, 'users')
project_nodes = get_nodes_from_partition(G, 'projects')

print(most_similar_users(G, 'u4560', user_nodes, project_nodes))


# Recommend
def recommend_repositories(G, from_user, to_user):
    # Get the set of repositories that from_user has contributed to
    from_repos = set(G.neighbors(from_user))
    # Get the set of repositories that to_user has contributed to
    to_repos = set(G.neighbors(to_user))

    # Identify repositories that the from_user is connected to that the to_user is not connected to
    return from_repos.difference(to_repos)

# Print the repositories to be recommended
print(recommend_repositories(G, 'u1570', 'u16091'))


#### GRAPH FROM PANDAS (METHOD 1)
import os
os.chdir('E:\Datacamp\Python\Network analysis\Part 2')

# Read in the data
import pandas as pd
revolution = pd.read_csv('american-revolution.csv')
revolution_melt = pd.melt(revolution, id_vars = 'Unnamed: 0')
revolution_melt_1 = revolution_melt[revolution_melt['value'] == 1 ]
edgelist = []
for index, row in revolution_melt_1.iterrows():
    edgelist.append((row[0], row[1]))
edgelist

# Read in the data
G = nx.Graph()
G.add_edges_from(edgelist)

# Assign nodes to 'clubs' or 'people' partitions
for n, d in G.nodes(data=True):
    if '.' in n:
        G.node[n]['bipartite'] = 'people'
    else:
        G.node[n]['bipartite'] = 'clubs'
        
# Print the edges of the graph
print(G.edges())


#### GRAPH FROM PANDAS (METHOD 2)
G = nx.Graph()

# Add nodes from each of the partitions
G.add_nodes_from(revolution_melt_1['Unnamed: 0'], bipartite = 'people')
G.add_nodes_from(revolution_melt_1['variable'], bipartite = 'clubs')

# Add in each edge along with the weight the edge was created
for index, row in revolution_melt_1.iterrows():
    G.add_edge(row['Unnamed: 0'], row['variable'], weight = row['value'])


#### COMPUTING PROJECTION OF A BIPARTITE GRAPH ON ONE PARTITION
# investigate(project) the relationships between nodes on one partition (normally, there would not be a connection between nodes in 1 partite)
    # conditioned on the connections to the nodes in the other partition
# example on p 3-6 chap 2: customers 1 and customer 2 all use product 2 => can be linked together ; only customer 3 uses product 1 => in an other group
# Prepare the nodelists needed for computing projections: people, clubs
people = [n for n in G.nodes() if G.node[n]['bipartite'] == 'people']
clubs = [n for n, d in G.nodes(data=True) if d['bipartite'] == 'clubs']

# Compute the people and clubs projections: peopleG, clubsG
peopleG = nx.bipartite.projected_graph(G, people)
clubsG = nx.bipartite.projected_graph(G, clubs)

peopleG.edges() # => new graph with new connection between peoples, removing all clubs
# This is the same as shared_partition_nodes functions above


#### DEGREE CENTRALITY ON PROJECTION
import matplotlib.pyplot as plt

# Plot the degree centrality distribution of both node partitions from the original graph
# Recall degree centrality definition: number of neighbors / number of possible neighbors
# This formula, denominator has changed to number of nodes on opposite partition
plt.figure() 
original_dc = nx.bipartite.degree_centrality(G, people)  
plt.hist(list(original_dc.values()), alpha=0.5)
plt.yscale('log')
plt.title('Bipartite degree centrality')
plt.show()
# has discrete values: this again stems from having only a small number of clubs nodes that the people nodes can connect to.

# Plot the degree centrality distribution of the peopleG graph
plt.figure()
people_dc = nx.degree_centrality(peopleG)
plt.hist(list(people_dc.values()))
plt.yscale('log')
plt.title('Degree centrality of people partition')
plt.show()
# peopleG histogram is more contiguous because of the large number of nodes

# Plot the degree centrality distribution of the clubsG graph
plt.figure()
clubs_dc = nx.degree_centrality(clubsG)
plt.hist(list(clubs_dc.values()))
plt.yscale('log')
plt.title('Degree centrality of clubs partition')
plt.show()
# The disjoint histogram from the clubsG graph stems from its small number of nodes (only 7)


#### COMPUTE ADJACENCY MATRIX
# Get the list of people and list of clubs from the graph: people_nodes, clubs_nodes
people_nodes = get_nodes_from_partition(G, 'people')
clubs_nodes = get_nodes_from_partition(G, 'clubs')

# Compute the biadjacency matrix: bi_matrix
bi_matrix = nx.bipartite.biadjacency_matrix(G, row_order=people_nodes, column_order=clubs_nodes)
bi_matrix.toarray()

# Compute the user-user projection: user_matrix
# By multiplying matrix with the tranposition, we eliminate the effect of 7 'clubs', the matrix now has shape of 254 x 254
user_matrix = bi_matrix @ bi_matrix.T # T for tranposition

print(user_matrix)


# IMPUTE METADATA BACK TO MATRIX
import numpy as np 

# Find out the names of people who were members of the most number of clubs
diag = user_matrix.diagonal()     # Returns the k-th diagonal of the matrix. (duong cheo ma tran)        
indices = np.where(diag == diag.max())[0] # np.where returns a tuple => access relevant indices by indexing to the tuple with [0]
print('Number of clubs: {0}'.format(diag.max()))
print('People with the most number of memberships:')
for i in indices:
    print('- {0}'.format(people_nodes[i]))  

# Set the diagonal to zero and convert it to a coordinate matrix format
user_matrix.setdiag(0)
users_coo = user_matrix.tocoo()

# Find pairs of users who shared membership in the most number of clubs
indices = np.where(users_coo.data == users_coo.data.max())[0] 
print('People with most number of shared memberships:')
for idx in indices:
    print('- {0}, {1}'.format(people_nodes[users_coo.row[idx]], people_nodes[users_coo.col[idx]])) 


#### MAKE NODELIST FOR PANDAS
# Initialize a list to store each edge as a record: nodelist
nodelist = []
for n, d in peopleG.nodes(data=True):
    # nodeinfo stores one "record" of data as a dict
    nodeinfo = {'person': n} 
    
    # Update the nodeinfo dictionary 
    nodeinfo.update(d)
    
    # Append the nodeinfo to the node list
    nodelist.append(nodeinfo)
    

# Create a pandas DataFrame of the nodelist: node_df
node_df = pd.DataFrame(nodelist)
print(node_df.head())


#### EDGELIST FOR PANDAS
# Initialize a list to store each edge as a record: edgelist
edgelist = []
for n1, n2, d in peopleG.edges(data=True):
    # Initialize a dictionary that shows edge information: edgeinfo
    edgeinfo = {'node1':n1, 'node2':n2}
    
    # Update the edgeinfo data with the edge metadata
    edgeinfo.update(d)
    
    # Append the edgeinfo to the edgelist
    edgelist.append(edgeinfo)
    
# Create a pandas DataFrame of the edgelist: edge_df
edge_df = pd.DataFrame(edgelist)
print(edge_df.head())