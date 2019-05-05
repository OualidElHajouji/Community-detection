"""
Purpose of the testing file:
Generates multiple BSM with different parameters c_in, c_out
The goal is to plot the quality depending on the graph ratio
Also, we evaluate the quality of clustering depending on the method used, and on the distribution.
"""

import numpy as np
import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 100
k = 2
cin = 90
cout = 75

print(qlt.graph_ratio(cin, cout, n))
rmax = np.sqrt(cin/np.log(n))  # Maximum value of the graph ratio


def misclustering_MonteCarloBasic(min = 0.2, max = 1.5, nb_X = 50, nb_Simu = 80, random = False, distribution = []):
    """Returns nb_X values of the quality of the clustering (accuracy), each one corresponding to a value of the graph_ratio """
    Quality = np.zeros((nb_Simu, nb_X))
    if(random):
        X = np.random.rand(nb_X)
    else:
        U = np.linspace(min, max, nb_X)   # Space of graph_ratio
        X = qlt.inverse(U, cin, n)        # Corresponding space of x = cout/cin
    classes = sbm.generateClasses(n, k, distribution)
    for j in range(nb_Simu):
        for i in range(nb_X):
            A = sbm.simulate(n, cin, X[i] * cin, k, classes)  # cout = x*cin
            clusters = spc.spectral_clustering(A, k, laplacian_method=0)
            Quality[j, i] = qlt.normal(classes, clusters, n)    #Quality for the current simulation, and current value of cout
        print(j)

    Y = np.mean(Quality, axis=0)
    plt.plot(U, Y)
    plt.xlabel("Graph ratio r")
    plt.ylabel("Q(r)")
    plt.show()

    return U, Y


def quality_method(nb_Simu = 1000, nb_bars = 5):
    """Plots an histogram of average quality over different distributions, for different methods"""
    Quality_0 = np.zeros((nb_Simu, nb_bars))   # For laplacian_method = 0
    Quality_1 = np.zeros((nb_Simu, nb_bars))   # For laplacian_method = 1

    distribution_list = [[0.5,0.5],[0.7,0.3], [0.8,0.2], [0.95,0.05], [0.99,0.01]]
    for j in range(nb_bars) :
        distribution = distribution_list[j]
        classes = sbm.generateClasses(n, k, distribution)
        for i in range(nb_Simu):
            print(i)
            A = sbm.simulate(n, cin, cout, k, classes)
            clusters = spc.spectral_clustering(A, k, laplacian_method=0)
            Quality_0[i,j]= qlt.normal(classes, clusters, n)
            clusters = spc.spectral_clustering(A, k, laplacian_method=0)
            Quality_1[i, j] = qlt.normal(classes, clusters, n)

    Y0 = np.mean(Quality_0, axis=0)
    Y1 = np.mean(Quality_1, axis=0)

    str_list = ['[0.5,0.5]','[0.7,0.3]', '[0.8,0.2]', '[0.95,0.05]', '[0.99,0.01]']
    plt.bar(np.arange(0, 45, 9), Y0, tick_label=str_list, label = "Without laplacian")
    plt.stem(np.arange(0, 45, 9) + 1, Y1, linefmt="r", markerfmt="ro", basefmt="None", label="With laplacian")
    plt.xlabel("Distribution", labelpad=15, fontsize=25)
    plt.ylabel("Quality", labelpad=40, fontsize=25)

    plt.xticks(fontsize=14)
    plt.legend(loc='best')
    plt.show()

    return()

quality_method()