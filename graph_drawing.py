import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""

Functions for Graph drawing

"""

def coloring(clusters, k):
    """Returns a list of colors corresponding to each node (depending on its cluster)"""
    n = np.size(clusters)
    if(k == 1):
        return np.array(['r']*n)
    if(k == 2):
        return np.array([int(clusters[i]==0)*'r' + int(clusters[i]==1)*'g' for i in range(n)])
    if(k == 3):
        return np.array([int(clusters[i]==0)*'r' + int(clusters[i]==1)*'g' + int(clusters[i]==2)*'b' for i in range(n)])

def matrixGraph(affinityMatrix, clusters, k):
    """Draws the graph given the clusters and the affinity matrix"""
    G = nx.from_numpy_matrix(affinityMatrix)
    pos = nx.spring_layout(G)
    G.to_undirected()

    color = coloring(clusters, k)

    nx.draw(G, node_color=color, node_size = 10, line_width = 0.1)
