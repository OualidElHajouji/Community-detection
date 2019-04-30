import numpy as np

def normal(classes, clusters, size):
    q = np.sum(classes == clusters)/size
    return max(q, 1-q)

def reEvaluate(classes, clusters, size):
    q = np.sum(classes == clusters)/size
    return (q > .8)*clusters + (q < .2)*(1-clusters)

def hyp_test(cin, cout, n):
    num = cin - cout
    num2 = np.sqrt(cin + cout)
    num3 = np.sqrt(np.log(n))
    return num/(num2*num3)

def inverse(r,cin,n):
    r1 = r*np.sqrt(np.log(n)/cin)
    x = ((2+r1**2)-r1*np.sqrt(r1**2+8))/2
    return x

def well_clustered(j, classes, clusters):
    return classes[j] == clusters[j]

def badly_clustered_test(vertices, cardinal_of_vertices, classes, clusters, test = 0):
    q = np.sum([int(classes[i] != clusters[i]) for i in vertices])
    return (1 - test)*(q != 0) + test*(q == cardinal_of_vertices)

"""
def mutateMatrix(A, classes, cin, cout, vertex):

// TO DO
"""

