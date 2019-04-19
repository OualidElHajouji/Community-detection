import numpy as np
import matplotlib.pyplot as plt
import networkx as nx




n = 1000
cin = 100
cout= 10
k = 2
def simulate(B,Classes):
    """Given a matrix B indicating probabilities of links between two elements of a class,
     simulation of the graph"""

    U = np.random.rand(n,n)
    U = (U + U.T)/2
    Matrix = np.array([[B[Classes[i],Classes[j]]*(i !=j) for i in range(n)] for j in range(n)])
    A = (U<=Matrix)
    return A

B = (cin/n-cout/n)*np.identity(k) + (cout/n)*np.ones((k,k))
p = k*[1/k]
Classes = np.random.choice(range(k), size = n, p = p)
color = np.array([Classes[i]*'r' + (1-Classes[i])*'b' for i in range(n)])

A = simulate(B, Classes)

G = nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)



G.to_undirected()
nx.draw(G, node_color=color)
plt.show()