"""

Con1tains the implementation of the basic clustering algorithm

Input: Similarity matrix (Here Adjacency matrix) noted A

Output: Two partitions L1, L2 representing the indices of the vertices belonging to each cluster respectively

Algorithm:
    Step 1: From the Adjacency matrix, generate one suitable Laplacian matrix
    Step 2: Select Laplacian eigenvectors corresponding to its top eigenvalues (top 2)
    Step 3: Run 2-means algorithm to these eigenvectors
"""

import numpy
import scipy
from sklearn.cluster import KMeans


def laplacian(A, laplacian_method = 1):
    """Calculates the laplacian matrix : two methods"""
    if laplacian_method==0:
      A = numpy.matrix.astype(A, float)
      return A
    if laplacian_method==1:
      D = numpy.zeros(A.shape)
      w = numpy.sum(A, axis=0)
      D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
      return D.dot(A).dot(D)
    if laplacian_method==2:
      D = numpy.zeros(A.shape)
      w = numpy.sum(A, axis = 0)
      D.flat[::len(w) + 1] = w ** (-1.)  # set the diag of D to w
      return D.dot(A)


def k_means(X, n_clusters):
    """Classical k_means algorithm"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
    return kmeans.fit(X).labels_


def spectral_clustering(affinity, n_clusters, cluster_method=k_means, laplacian_method=1):
    """Main spectral clustering algorithm"""
    L = laplacian(affinity, laplacian_method)
    eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
    X = eig_vect.real
    rows_norm = numpy.linalg.norm(X, axis=1, ord=2)
    Y = (X.T / rows_norm).T
    labels = cluster_method(Y, n_clusters)
    return labels
