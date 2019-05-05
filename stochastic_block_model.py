"""
Simulates the Block Stochastic Model

Contains the parameters of the test
"""

import numpy as np

def generateClasses(n, k, distribution = []):
    """Returns a list that associates each node to its class, with a given distribution"""
    if distribution == []:  # Case with uniform distribution
        p = k * [1 / k]
        return np.random.choice(range(k), size=n, p=p)
    else:  #Case with given distribution
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
     we simulate the graph according to the Stochastic Block Model"""
    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))  # Matrix B described in instructions

    U = np.random.rand(n,n)
    U = (U + U.T)/2  # In order to get an undirected graph at the end (i.e symmetrical adjacency matrix)
    Matrix = np.array([[B[Classes[i],Classes[j]]*(i !=j) for i in range(n)] for j in range(n)])
    A = (U<=Matrix)
    return A


def simulate_Importance_Sampling(n, cin, cout, k, set_of_vertices, Classes, model):
    """Given a matrix B indicating probabilities of links between two elements of two classes, we simulate the graph with a slightly modified SBM. If model = 0, we invert,
    for the elements of set, the probabilites of intra-class and extra-class links, then simulate the graph. Else, if model = 1, we set every probability to 0.5 for i in set."""

    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))
    U = np.random.rand(n, n)
    U = (U + U.T) / 2
    matrix = np.array([[B[Classes[i], Classes[j]] * (i != j) for i in range(n)] for j in range(n)])  # We follow the SBM at the beginning
    modified_matrix = np.copy(matrix)
    for i in set_of_vertices:
        for j in range(n):
            modified_matrix[i,j] = (1-model)*((cout/n)*int((matrix[i,j]==cin/n)) + (cin/n)*int((matrix[i,j]==cout/n)))+(model*(1/2)*(i!=j)) #Inversion of the probabilities if model =0, probability = 0.5 else.
            modified_matrix[j,i] = modified_matrix[i,j]

    A = (U <= modified_matrix)
    return A

