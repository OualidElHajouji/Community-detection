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

def equivalence_classes(classes,k):
    """Returns a list of length k such that each cell i contains the list of the elements that belong to the class k i"""
    n = len(classes)
    list_of_classes = [[]]*k
    for i in range(n):
            list_of_classes[classes[i]].append(i)
    return list_of_classes

def simulate_Importance_Sampling(n, cin, cout, k, distribution = [], set):
    """Given a matrix B indicating probabilities of links between two elements of two classes, we invert, for the elements of set, the probabilites
     of intra-class and extra-class links, then simulate the graph"""
    B = (cin / n - cout / n) * np.identity(k) + (cout / n) * np.ones((k, k))
    p = k * [1 / k]
    Classes = generateClasses(n, k, distribution)

    U = np.random.rand(n, n)
    U = (U + U.T) / 2
    Matrix = np.array([[B[Classes[i], Classes[j]] * (i != j) for i in range(n)] for j in range(n)])

    for i in set:
        for j in range(n):
            Matrix[i,j]=(cout/n)*(Matrix[i,j]==cin/n) + (cin/n)*(Matrix[i,j]==cin/n)
            Matrix[j,i]=Matrix[i,j]
    A = (U <= Matrix)
    return A, Classes