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

def simulate(n, cin, cout, k, Classes):
    """Given a matrix B indicating probabilities of links between two elements of two classes,
     simulation of the graph"""
    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))
    p = k * [1 / k]

    U = np.random.rand(n,n)
    U = (U + U.T)/2
    Matrix = np.array([[B[Classes[i],Classes[j]]*(i !=j) for i in range(n)] for j in range(n)])
    A = (U<=Matrix)
    return A


def simulate_Importance_Sampling(n, cin, cout, k, set, Classes):
    """Given a matrix B indicating probabilities of links between two elements of two classes, we invert, for the elements of set, the probabilites
     of intra-class and extra-class links, then simulate the graph"""
    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))
    U = np.random.rand(n, n)
    U = (U + U.T) / 2
    matrix = np.array([[B[Classes[i], Classes[j]] * (i != j) for i in range(n)] for j in range(n)])
    modified_matrix = np.copy(matrix)
    for i in set:
        for j in range(n):
            modified_matrix[i,j]=(cout/n)*int((matrix[i,j]==cin/n)) + (cin/n)*int((matrix[i,j]==cout/n))
            modified_matrix[j,i] = modified_matrix[i,j]
    A = (U <= modified_matrix)
    return A


