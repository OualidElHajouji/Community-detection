"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
"""

import numpy as np
import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import clustering_quality as qlt

n = 100
k = 2
cin = 100
cout = 10


def misclustering_MonteCarloBasic(nb_X, nb_Simu, random = False):
    Quality = np.array([[-1 for array in range(nb_X)]])
    if(random):
        X = np.random.rand(nb_X)
    else:
        X = np.array([i for i in range(nb_X+1)])/(nb_X+1)
    for j in range(nb_Simu):
        row = np.array([])
        for i in range(1, nb_X + 1):
            A, classes = bsm.simulate(n, cin, X[i] * cin, k)
            clusters = spc.spectral_clustering(A, k)
            row = np.append(row, qlt.normal(classes, clusters, n))
        Quality = np.append(Quality, [row], 0)
        print(row)
    return Quality

