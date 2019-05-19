"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
"""

import numpy as np
import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 200
k = 2
cin = 100
cout = 10

rmax = np.sqrt(cin/np.log(n))


def misclustering_MonteCarloBasic(min = 0.2, max = 1, nb_X = 80, nb_Simu = 80, laplacian = 0, random=False, distribution=[]):
    Quality = np.zeros((nb_Simu, nb_X))
    if(random):
        X = np.random.rand(nb_X)
    else:
        U = np.linspace(min, max, nb_X)
        X = qlt.inverse(U, cin, n)
        print(X)
    classes = bsm.generateClasses(n, k, distribution=distribution)
    for j in range(nb_Simu):
        for i in range(nb_X):
            A= bsm.simulate(n, cin, X[i] * cin, k, classes)
            clusters = spc.spectral_clustering(A, k, laplacian_method= laplacian)
            Quality[j, i] = qlt.normal(classes, clusters, n)
        print(j)
    return U, np.mean(Quality, axis=0)

"""
PERFORMED TESTS FOR THE QUALITY CURVE
X1, Y1 = misclustering_MonteCarloBasic()
plt.plot(X1, Y1, color='b', label='Quality Q without Laplacian')

X2, Y2 = misclustering_MonteCarloBasic(laplacian=2)
plt.plot(X2, Y2, color='r', label='Quality Q for division Laplacian')

X3, Y3 = misclustering_MonteCarloBasic(laplacian=1)
plt.plot(X3, Y3, color='g', label='Quality Q for symmetrical division Laplacian')
"""

plt.xlabel("Graph ratio r")
plt.ylabel("Clustering quality Q")
plt.title("Curve of the quality Q in terms of the graph ratio r in case of distribution [0.3, 0.7]", fontsize=13)
plt.legend(loc="best", fontsize=13)

plt.show()



