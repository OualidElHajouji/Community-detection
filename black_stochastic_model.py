"""
Simulates the Block Stochastic Model

Contains the parameters of the test
"""

import numpy as np

def generateClasses(n, k, distribution = []):
    p = k * [1 / k]
    if distribution == []:
        return np.random.choice(range(k), size=n, p=p)
    else:
        res = np.zeros(n, dtype=int)
        for i in range(n):
            r = np.random.rand()
            j = 0
            s = distribution[0]
            while(s < r):
                j += 1
                s += distribution[j]
            res[i] = j
        return res

def simulate(n, cin, cout, k, distribution = []):
    """Given a matrix B indicating probabilities of links between two elements of a class,
     simulation of the graph"""
    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))
    p = k * [1 / k]
    Classes = generateClasses(n, k, distribution)

    U = np.random.rand(n,n)
    U = (U + U.T)/2
    Matrix = np.array([[B[Classes[i],Classes[j]]*(i !=j) for i in range(n)] for j in range(n)])
    A = (U<=Matrix)
    return A, Classes
