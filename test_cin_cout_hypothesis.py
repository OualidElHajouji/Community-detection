"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
"""

import numpy as np
import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 100
k = 2
cin = 90
cout = 10

rmax = np.sqrt(cin/np.log(n))


def misclustering_MonteCarloBasic(min = 0.2, max = 1.5, nb_X = 50, nb_Simu = 80, random = False):
    Quality = np.zeros((nb_Simu, nb_X))
    if(random):
        X = np.random.rand(nb_X)
    else:
        U = np.linspace(min, max, nb_X)
        X = qlt.inverse(U, cin, n)
        print(X)
    classes = bsm.generateClasses(n, k, distribution = [])
    for j in range(nb_Simu):
        for i in range(nb_X):
            A= bsm.simulate(n, cin, X[i] * cin, k, classes)
            clusters = spc.spectral_clustering(A, k, laplacian_method=0)
            Quality[j, i] = qlt.normal(classes, clusters, n)
        print(j)
    return U, np.mean(Quality, axis=0)

X, Y = misclustering_MonteCarloBasic()
plt.plot(X, Y)
plt.show()



