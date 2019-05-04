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

n = 100
cin = 90
cout = 72
k = 2
distribution = [.2, .8]

print(qlt.hyp_test(cin, cout, n))

def monteCarloApproach(nb_Simu = 10000, set_of_vertices = range(1)):
    res = 0
    classes = bsm.generateClasses(n, k, distribution = [])
    v = len(set_of_vertices)
    classes = [0]*50+[1]*50
    for i in range(nb_Simu):
        print(i)
        graph = bsm.simulate(n, cin, cout, k, classes)
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(graph, k), n)
        res += qlt.badly_clustered_test(set_of_vertices, v, classes, clusters)
    return res/nb_Simu


def equivalence_classes(classes):
    """ Returns the list (of length k) of all classes"""
    classes_list = [[]]*k
    for i in range(len(classes)):
        classes_list[classes[i]] = classes_list[classes[i]] +[i]
    return classes_list

def exponent_inside(graph, classes, classes_list, set_of_vertices):
    res = 0
    for i in set_of_vertices:
        classe = classes_list[classes[i]]
        for j in classe:
            res += 2*graph[i,j]
        res -= np.sum(graph[i,:])
    return res

def exponent_outside(classes, classes_list, set_of_vertices):
    sum = 0
    for i in set_of_vertices:
        sum += 2*len(classes_list[classes[i]])-n-1
    return sum

def importanceSamplingApproach(nb_Simu = 10000, set_of_vertices = range(1)):
    res = 0
    classes = bsm.generateClasses(n, k, distribution=[])
    classes =[0]*50 + [1]*50
    classes_list = equivalence_classes(classes)
    pin = cin/n
    pout = cout/n
    x = (pin*(1-pout))/(pout*(1-pin))
    print(x)
    v = len(set_of_vertices)
    for i in range(nb_Simu):
        print(i)
        graph = bsm.simulate_Importance_Sampling(n, cin, cout, k, set_of_vertices, classes)
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(graph, k), n)
        badly_clustered = qlt.badly_clustered_test(set_of_vertices, v, classes, clusters)
        exp_inside = exponent_inside(graph, classes, classes_list, set_of_vertices)
        res += np.power(x,exp_inside)*int(badly_clustered)
    exp_outside = exponent_outside(classes, classes_list, set_of_vertices)
    res = (res*np.power((1-pin)/(1-pout),exp_outside))/nb_Simu
    return res


print(importanceSamplingApproach(nb_Simu = 10000, set_of_vertices = range(1)))
#print(monteCarloApproach(nb_Simu = 10000, set_of_vertices = range(1)))


