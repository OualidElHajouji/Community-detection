import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

B = (cin/n-cout/n)*np.identity(k) + (cout/n)*np.ones((k,k))
p = k*[1/k]
Classes = np.random.choice(range(k), size = n, p = p)
spc.spectral_clustering(A, 2)


color = np.array([Classes[i]*'r' + (1-Classes[i])*'b' for i in range(n)])

A = bsm.simulate(B, Classes)

G = nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)



G.to_undirected()
nx.draw(G, node_color=color)
plt.show()