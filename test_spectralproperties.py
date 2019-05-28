"""
Purpose of the testing file:
1-View the spectral properties of different Laplacian matrices
2-Compare the different specters

"""


import numpy as np
import scipy as scp
import spectral_clustering_1 as spc
import stochastic_block_model as bsm
import matplotlib.pyplot as plt

def generateEigenElements(cin=200, cout=50, n=400, k=2, distribution=[], laplacian=0):
    classes = bsm.generateClasses(n, k, distribution)
    A = bsm.simulate(n, cin, cout, k, classes)
    L = spc.laplacian(A, laplacian)

    eig_val, eig_vect = scp.sparse.linalg.eigs(L)
    return eig_val, eig_vect

"""
Preparing the eigenvalues
"""
eig_val0, eig_vect0 = generateEigenElements(laplacian=0)
Y0 = np.sort(eig_val0.real)
X0 = np.arange(np.size(Y0))

eig_val1, eig_vect1 = generateEigenElements(laplacian=1)
Y1 = np.sort(eig_val1.real)
X1 = np.arange(np.size(Y1))

eig_val2, eig_vect2 = generateEigenElements(laplacian=2)
Y2 = np.sort(eig_val2.real)
X2 = np.arange(np.size(Y2))

"""
Ploting everything
"""
plt.figure(0)
plt.xlabel('Eigenvalues (ordered)')
plt.ylabel('Value')
plt.title('Comparing left figure eigenvalues to Adjacency matrix spectrum')

plt.plot(X0, Y0, 'ro', label='Eigenvalues for Adjacency matrix')
plt.plot(X1-.05, Y1, 'bo', label='Eigenvalues for symmetrical division')
plt.plot(X2+.05, Y2, 'go', label='Eigenvalues for non-symmetrical division')

plt.legend(loc="best", fontsize=7)

plt.plot()

plt.show()