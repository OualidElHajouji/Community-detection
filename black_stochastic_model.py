"""
Simulates the Block Stochastic Model

Contains the parameters of the test
"""

import numpy as np

def simulate(n, cin, cout, k):
    """Given a matrix B indicating probabilities of links between two elements of a class,
     simulation of the graph"""
    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))
    p = k * [1 / k]
    Classes = np.random.choice(range(k), size=n, p=p)

    U = np.random.rand(n,n)
    U = (U + U.T)/2
    Matrix = np.array([[B[Classes[i],Classes[j]]*(i !=j) for i in range(n)] for j in range(n)])
    A = (U<=Matrix)
    return A