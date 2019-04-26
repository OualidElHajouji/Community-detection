import numpy as np

def normal(classes, clusters, size):
    q = np.sum([int(classes[i] == clusters[i])/size for i in range(size)])
    return max(q, 1-q)

def hyp_test(cin, cout, n):
    return np.sqrt(cin - cout)/np.log(n)