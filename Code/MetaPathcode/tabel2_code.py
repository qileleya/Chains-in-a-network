import pandas as pd
import networkx as nx
import random

# Load the investment data
df = pd.read_csv('C:/Users/14522/Desktop/博士idea/zip/link_prediction/graph/touzi/投资数据集/投资行为数据（2015-2020）.txt', sep='\t', header=None, names=['Investor', 'Type', 'Project'])
filtered_df = df[df['Type'] >= 7249][:500]
# Create a graph
G = nx.Graph()

# Add edges to the graph
for _, row in filtered_df.iterrows():
    investor = f"I_{row['Investor']}"
    t_ype = f"T_{row['Type']}"
    project = f"P_{row['Project']}"
    G.add_edge(investor, t_ype)
    G.add_edge(t_ype, project) 
    G.add_edge(investor, project)
    G.add_edge(investor, investor)


# Function to perform random walks
def random_walk(G, start_node, walk_length):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk

# Perform random walks
walks = []
for node in G.nodes(): 
    for _ in range(10):  # Number of walks per node
        walks.append(random_walk(G, node, walk_length=5))

# Function to compute metapath counts
def compute_metapath_counts(walks):
    metapath_counts = {
        'IT': 0,
        'IP': 0,
        'IPI': 0,
        'PT': 0,
        'IPT': 0
    }
    
    for walk in walks:
        for i in range(len(walk) - 1):
            if walk[i].startswith('I_') and walk[i+1].startswith('T_'):
                metapath_counts['IT'] += 1
            if walk[i].startswith('I_') and walk[i+1].startswith('P_'):
                metapath_counts['IP'] += 1
            if i < len(walk) - 2 and walk[i].startswith('I_') and walk[i+1].startswith('P_') and walk[i+2].startswith('I_'):
                metapath_counts['IPI'] += 1
            if walk[i].startswith('P_') and walk[i+1].startswith('T_'):
                metapath_counts['PT'] += 1
            if i < len(walk) - 2 and walk[i].startswith('I_') and walk[i+1].startswith('P_') and walk[i+2].startswith('T_'):
                metapath_counts['IPT'] += 1

    return metapath_counts

# Compute the metapath counts
metapath_counts = compute_metapath_counts(walks)

# Print the results
for metapath, count in metapath_counts.items():
    print(f"Metapath {metapath}: {count} occurrences")