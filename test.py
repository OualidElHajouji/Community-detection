"""
Initial testing. Purpose:
-Tests visually the correction of the BSM implementation

-Tests the correction of the spectral clustering algorithm

-Gives a sketch of further tests files used in the project

"""

import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import matplotlib.pyplot as plt
import graph_drawing as gd

" Phase 1: Initialise the parameters of the test"
n = 100
cin = 90
cout = 10
k = 2

classesA = sbm.generateClasses(n, k, distribution=[])
A = sbm.simulate_Importance_Sampling(n, cin, cout, k, range(1),classesA)

" Phase 2: Apply Spectral Clustering "
labelsA = spc.spectral_clustering(A, k, laplacian_method=0)

" Phase 3: Visualise well the data in a suitable graph "
gd.matrixGraph(A, labelsA, k)


plt.show()