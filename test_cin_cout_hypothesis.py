"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
"""

import numpy as np
import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 150
k = 2
cin = 100
cout = 10

rmax = np.sqrt(cin/np.log(n))


def misclustering_MonteCarloBasic(nb_X, nb_Simu, random = False):
    Quality = np.zeros((nb_Simu, nb_X))
    if(random):
        X = np.random.rand(nb_X)
    else:
        U = np.linspace(0.3, 0.5, nb_X)
        X = qlt.inverse(U, cin, n)
        print(X)
    for j in range(nb_Simu):
        for i in range(nb_X):
            A, classes = bsm.simulate(n, cin, X[i] * cin, k)
            clusters = spc.spectral_clustering(A, k)
            Quality[j, i] = qlt.normal(classes, clusters, n)
        print(Quality[j])
    return U, np.mean(Quality, axis=0)

X, Y = misclustering_MonteCarloBasic(40, 50)
plt.plot(X, Y)

plt.show()



