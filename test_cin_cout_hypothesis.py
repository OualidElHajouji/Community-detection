"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
"""

import numpy as np
import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import clustering_quality as qlt

n = 1000
k = 2
cin = 100
cout = 10

A_Simulations = np.array([])
A_Classes = np.array([])
Quality = np.array([])
for j in range(10):
   print("----------------")
   for i in range(1, 11):
        A, classes = bsm.simulate(n, cin, i*cout, k)
        C = spc.spectral_clustering(A, k)

        quality = qlt.normal(classes, C, n)
        print(quality)