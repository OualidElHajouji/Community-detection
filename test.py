import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import matplotlib.pyplot as plt
import graph_drawing as gd

n = 1000
cin = 100
cout = 10
k = 2

A = bsm.simulate(n, cin, cout, k)
labels = spc.spectral_clustering(A, k)

gd.matrixGraph(A, labels, k)
plt.show()