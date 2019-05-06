"""

Purpose of the test file:
Given a certain subset of vertex within a random graph, estimate the probability of
clustering badly the elements of this subset

Considered a rare event.

Approach 1: Using a basic Monte-Carlo estimation. Expected probability: 0.
Approach 2: Using importance sampling. Expected probability: >0.

"""

import numpy as np
import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 100
cin = 90
cout = 50
k = 2


print(qlt.graph_ratio(cin, cout, n))


def monteCarloApproach(nb_Simu=10000, set_of_vertices=range(1), random_class=False):
    """Calculates the probability of misclustering with a naive Monte Carlo approach"""
    res = 0
    if (random_class):   #The class can either be random, or chosen
        classes = sbm.generateClasses(n, k, distribution=[])
    else:
        classes = [0] * (n // 2) + [1] * (n - n // 2)
    v = len(set_of_vertices)

    for i in range(nb_Simu):
        print(i)
        graph = sbm.simulate(n, cin, cout, k, classes)
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(graph, k), n)   #We make sure to get the labels of the clusters right
        res += qlt.badly_clustered_test(set_of_vertices, v, classes, clusters)
    return res / nb_Simu


def equivalence_classes(classes):
    """ Returns the list (of length k) of all classes"""
    classes_list = [[]] * k
    for i in range(len(classes)):
        classes_list[classes[i]] = classes_list[classes[i]] + [i]
    return classes_list


def subset_superior_to(set, i):
    """ Returns the subset of set that contains elements superior to i"""
    res = []
    for j in set:
        if (j > i):
            res.append(j)
    return res


def factor_inside(graph, classes, classes_list, set_of_vertices, pin, pout, model):
    """Calculates the factor inside the expectation in the importance sampling formula, depending on the model """
    if (model==0):  # Probability inversion
        for i in set_of_vertices:
            classe = subset_superior_to(classes_list[classes[i]],i)   # All vertices related to i and > i
            sum = 2*np.sum(graph[i,classe])-np.sum(graph[i, i+1:])
        return np.power((pin*(1-pout))/(pout*(1-pin)), sum)  #We apply the theoretical formula
    else:  # 0.5-model
        product = 1
        for i in set_of_vertices:
            classe = subset_superior_to(classes_list[classes[i]],i)
            sum = np.sum(graph[i,classe])
            product = product*np.power((pin/(1-pin)), sum)*np.power((pout/(1-pout)), (np.sum(graph[i,i+1:])-sum))
        return product


def factor_outside(classes, classes_list, set_of_vertices, pin, pout, model):
    """Calculates the factor outside the expectation in the importance sampling formula, depending on the model """
    if model == 0:
        sum = 0
        for i in set_of_vertices:
            classe = subset_superior_to(classes_list[classes[i]], i)
            sum += 2 * len(classe) - n + i+1
        return np.power((1 - pin) / (1 - pout), sum)
    else:
        product = 1
        for i in set_of_vertices:
            classe = subset_superior_to(classes_list[classes[i]], i)
            product *= np.power(2.,n-i-1)*np.power(1-pin,len(classe))*np.power(1-pout,n-i-1-len(classe))
        return product

def importanceSamplingApproach(nb_Simu=10000, set_of_vertices=range(1), model=0, random_class=False):
    """Returns the probability of misclustering of a given set_of_vertices using a model of importance sampling"""
    res = 0

    if random_class:
        classes = sbm.generateClasses(n, k, distribution=[])
    else:
        classes = [0] * (n // 2) + [1] * (n - n // 2)

    classes_list = equivalence_classes(classes)
    pin = cin / n
    pout = cout / n
    v = len(set_of_vertices)

    for i in range(nb_Simu):
        print(i)  #In order to see the progress of the algorithm

        graph = sbm.simulate_Importance_Sampling(n, cin, cout, k, set_of_vertices, classes, model)  # Simulation specific to the importance sampling model
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(graph, k), n)
        badly_clustered = qlt.badly_clustered_test(set_of_vertices, v, classes, clusters)  # Whether the vertices are badly clustered
        fac_inside = factor_inside(graph, classes, classes_list, set_of_vertices, pin, pout, model)
        res += fac_inside * int(badly_clustered)    # This is inside the expectation


    fac_outside = factor_outside(classes, classes_list, set_of_vertices, pin, pout, model) # Outside the expectation
    res = (res * fac_outside) / nb_Simu
    return res


print(importanceSamplingApproach(nb_Simu=10000, set_of_vertices=range(1), model=0))

#print(monteCarloApproach(nb_Simu = 1000, set_of_vertices = range(1)))

