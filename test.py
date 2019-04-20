"""
Initial testing. Purpose:
-Tests visually the correction of the BSM implementation

-Tests the correction of the spectral clustering algorithm

-Gives a sketch of further tests files used in the project

"""

import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import matplotlib.pyplot as plt
import graph_drawing as gd

" Phase 1: Initialise the parameters of the test"
n = 600
cin = 100
cout = 10
k = 3


A, classesA = bsm.simulate(n, cin, cout, k)

" Phase 2: Apply Spectral Clustering "
labelsA = spc.spectral_clustering(A, k)

" Phase 3: Visualise well the data in a suitable graph "
gd.matrixGraph(A, labelsA, k)

plt.show()