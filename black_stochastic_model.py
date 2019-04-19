"""
Simulates the Block Stochastic Model

Contains the parameters of the test
"""

import numpy as np

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