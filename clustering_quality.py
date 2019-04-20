import numpy as np

def normal(classes, clusters, size):
    return np.sum([int(classes[i] == clusters[i]) for i in range(size)])