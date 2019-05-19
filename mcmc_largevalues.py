"""Purpose:
   Evaluate the probability of badly clustering a large set of vertices (n/4), with a graph ratio sufficiently low.
   The method used is MCMC simulation. We split the problem into K sub-problems, build Markov Chains for each sub-problem and
   evaluate a probability for each step with Monte-Carlo."""


import numpy as np
import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 20
cin = 7
cout = 5
k = 2

m = n//4  # Length of the set of vertices


def next_Markov_Graph(initial_graph, classes,p):
    """Returns next potential step in Markov Chain"""
    new_graph = sbm.simulate(n,cin,cout,k,classes)
    U = np.random.rand(n,n)
    change = U <= p  #Indicates which edges are to be modified
    return(new_graph*change + initial_graph*(1-change))

def accept(a, classes, clusters, set_of_vertices):
    """Returns whether we accept a graph as a Markov chain step or not, given a minimum number of misclustered vertices"""
    return(int(qlt.nb_of_misclustered(set_of_vertices,classes,clusters) >=a))

def MCMC_simulation(p=0.6,M = 1000,set_of_vertices = range(m),K = m, bound_split = np.linspace(0,m,m+1)):

    intermediate_probas = np.zeros((M,K))
    classes = [0] * (n // 2) + [1] * (n - n // 2)
    initial_graph = sbm.simulate(n,cin,cout,k,classes)

    for i in range(K):
        print(i)
        a = bound_split[i]
        b = bound_split[i + 1]
        current_graph = initial_graph
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(current_graph, k), n)
        intermediate_probas[0,i]= qlt.nb_of_misclustered(set_of_vertices,classes,clusters) >=b

        for j in range(1,M):

            new_graph = next_Markov_Graph(current_graph,classes,p)
            new_clusters = qlt.reEvaluate(classes, spc.spectral_clustering(new_graph, k), n)
            accepted = accept(a, classes, new_clusters, set_of_vertices)
            current_graph = accepted * new_graph + (1 - accepted) * current_graph

            if (not accepted):
                intermediate_probas[j,i]=intermediate_probas[j-1,i]
            else :
                intermediate_probas[j, i] = qlt.nb_of_misclustered(set_of_vertices, classes, new_clusters) >= b
                if intermediate_probas[j,i]:
                    initial_graph = current_graph

    intermediate_probas = np.mean(intermediate_probas, axis = 0)

    return (np.prod(intermediate_probas))

print(MCMC_simulation())