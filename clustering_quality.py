"""

Various functions for evaluating the quality of clustering

"""

import numpy as np

def normal(classes, clusters, size):
    """Returns proportion of well-clustered nodes"""
    q = np.sum(classes == clusters)/size
    return max(q, 1-q)

def reEvaluate(classes, clusters, size):
    """Solves the issue of having clusters with the wrong labels. Only works with k = 2"""
    q = np.sum(classes == clusters)/size
    return (q > .5)*clusters + (q <= .5)*(1-clusters)

def graph_ratio(cin, cout, n):
    """Returns the graph_ratio"""
    num = cin - cout
    num2 = np.sqrt(cin + cout)
    num3 = np.sqrt(np.log(n))
    return num/(num2*num3)

def inverse(r,cin,n):
    """Finds the x = cout/cin corresponding to a given graph_ratio r"""
    r1 = r*np.sqrt(np.log(n)/cin)
    x = ((2+r1**2)-r1*np.sqrt(r1**2+8))/2
    return x

def well_clustered(j, classes, clusters):
    """Is j in the right cluster ?"""
    return classes[j] == clusters[j]

def badly_clustered_test(vertices, cardinal_of_vertices, classes, clusters, test = 1):
    """Says whether a set of vertices is badly clustered. test = 0 --> well-clustered = at least one node is badly clustered
                                                          test = 1 --> well-clustered = every node is badly clustered"""
    q = np.sum([int(classes[i] != clusters[i]) for i in vertices])
    return (1 - test)*(q != 0) + test*(q == cardinal_of_vertices)

def nb_of_misclustered(set_of_vertices, classes, clusters):
    """Returns the number of misclustered vertices among the set of vertices"""
    return(np.sum([int(classes[i] != clusters[i]) for i in set_of_vertices]))


