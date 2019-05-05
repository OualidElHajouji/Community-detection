"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
"""

import numpy as np
import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 100
k = 2
cin = 90
cout = 10

rmax = np.sqrt(cin/np.log(n))  # Maximum value of the graph ratio


def misclustering_MonteCarloBasic(min = 0.2, max = 1.5, nb_X = 50, nb_Simu = 80, random = False, distribution = []):
    """Returns nb_X values of the quality of the clustering (accuracy), each one corresponding to a value of the graph_ratio """
    Quality = np.zeros((nb_Simu, nb_X))
    if(random):
        X = np.random.rand(nb_X)
    else:
        U = np.linspace(min, max, nb_X)   # Space of graph_ratio
        X = qlt.inverse(U, cin, n)        # Corresponding space of x = cout/cin
    classes = sbm.generateClasses(n, k, distribution)
    for j in range(nb_Simu):
        for i in range(nb_X):
            A = sbm.simulate(n, cin, X[i] * cin, k, classes)  # cout = x*cin
            clusters = spc.spectral_clustering(A, k, laplacian_method=0)
            Quality[j, i] = qlt.normal(classes, clusters, n)    #Quality for the current simulation, and current value of cout
        print(j)
    return U, np.mean(Quality, axis=0)

X, Y = misclustering_MonteCarloBasic()
plt.plot(X, Y)
plt.show()



