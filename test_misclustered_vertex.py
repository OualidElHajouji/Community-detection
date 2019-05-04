"""

Purpose of the test file:
Given a certain subset of vertex within a random graph verifying the ratio hypothesis, estimate the probability of
clustering badly the elements of this subset

Considered a rare event.

Approach 1: Using a basic Monte-Carlo estimation. Expected probability: 0.
Approach 2: Using importance sampling. Expected probability: >0.

"""

import numpy as np
import spectral_clusering_1 as spc
import black_stochastic_model as bsm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 400
cin = 200
cout = 20
k = 2
distribution = [.2, .8]

print(qlt.hyp_test(cin, cout, n))

def monteCarloApproach(nb_Simu = 100, set_of_vertices = range(200), test = 0):
    res = 0
    classes = generateClasses(n, k, distribution = [])
    for i in range(nb_Simu):
        graph = bsm.simulate(n, cin, cout, k, distribution)
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(graph, k), n)
        res += qlt.badly_clustered_test(set_of_vertices, v, classes, clusters, test)
    return res/nb_Simu


TO DO:
def importanceSamplingApproach(nb_Simu = 100, set_of_vertices = range(200), set, test = 0):

