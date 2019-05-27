"""

Purpose of the test file:
Given a certain subset of vertex within a random graph, estimate the probability of
clustering badly the elements of this subset. In all this part, the subset contains one element,
even though the functions are scalable and can be used for any subset.

Considered a rare event.

Approach 1: Using a basic Monte-Carlo estimation. Expected probability: 0.
Approach 2: Using importance sampling. Expected probability: >0. We have 2 models of importance sampling:
                                                                 Model 0: Probability inversion
                                                                 Model 1: General model

"""

import numpy as np
import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import clustering_quality as qlt
import matplotlib.pyplot as plt

n = 100
cin = 90
cout = 72
k = 2


print(qlt.graph_ratio(cin, cout, n))


def monteCarloApproach(nb_Simu=10000, set_of_vertices=range(1), random_class=False):
    """Calculates the probability of misclustering with a naive Monte Carlo approach"""
    res = 0
    if (random_class):   #The class can either be random, or chosen
        classes = sbm.generateClasses(n, k, distribution=[])
    else:
        classes = [0] * (n // 2) + [1] * (n - n // 2)   # Codominant classes [0,0,0...0,1...1,1,1]
    v = len(set_of_vertices)

    for i in range(nb_Simu):
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


def factor_inside(graph, classes, classes_list, set_of_vertices, pin, pout, qin, qout, model):
    """Calculates the factor inside the expectation in the importance sampling formula, depending on the model (model 0 : inversion, model 1 : general with parameters qin,qout"""
    if (model==0):  # Probability inversion (qin =pout, qout = pin)
        sum = 0
        for i in set_of_vertices:
            classe = subset_superior_to(classes_list[classes[i]],i)   # All vertices related to i and > i
            sum += 2*np.sum(graph[i,classe])-np.sum(graph[i, i+1:])
        return np.power((pin*(1-pout))/(pout*(1-pin)), sum)  #We apply the theoretical formula
    else:  # model with given probabilities qin, qout
        product = 1
        for i in set_of_vertices:
            classe = subset_superior_to(classes_list[classes[i]],i)
            sum = np.sum(graph[i,classe])
            product = product*np.power((pin*(1-qin))/(qin*(1-pin)), sum)*np.power((pout*(1-qout))/(qout*(1-pout)), (np.sum(graph[i,i+1:])-sum))
        return product


def factor_outside(classes, classes_list, set_of_vertices, pin, pout, qin, qout, model):
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
            product *= np.power((1-pin)/(1-qin),len(classe))*np.power((1-pout)/(1-qout),n-i-1-len(classe))
        return product

def importanceSamplingApproach(qin, qout,nb_Simu=10000, set_of_vertices=range(1), model=1, random_class=False):
    """Returns the probability of misclustering of a given set_of_vertices using a model (0 or 1) of importance sampling"""
    simulation = np.zeros(nb_Simu)
    if random_class:
        classes = sbm.generateClasses(n, k, distribution=[])
    else:
        classes = [0] * (n // 2) + [1] * (n - n // 2)

    classes_list = equivalence_classes(classes)   #List of classes
    pin = cin / n
    pout = cout / n
    v = len(set_of_vertices)

    for i in range(nb_Simu):

        graph = sbm.simulate_Importance_Sampling(n, cin, cout, qin, qout, k, set_of_vertices, classes, model)  # Simulation specific to the importance sampling model (with probability change)
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(graph, k), n)  #Clustering
        badly_clustered = qlt.badly_clustered_test(set_of_vertices, v, classes, clusters)  # Whether the vertices are badly clustered
        fac_inside = factor_inside(graph, classes, classes_list, set_of_vertices, pin, pout, qin, qout, model)
        simulation[i]= fac_inside * int(badly_clustered)    # This is inside the expectation

    fac_outside = factor_outside(classes, classes_list, set_of_vertices, pin, pout,qin,qout, model) # Outside the expectation
    simulation = fac_outside*simulation
    return simulation

def test_importance_sampling(nb_Simu):
    """Computes and plots all the results, for given parameters, of importance sampling algorithm"""
    simulation = importanceSamplingApproach(0.2,0.9,nb_Simu=nb_Simu, set_of_vertices=range(25), model=0)
    estimator = np.cumsum(simulation)/(np.arange(1,nb_Simu+1))   #Estimator of probability
    proba = estimator[-1]
    std = np.std(simulation)
    a = max(proba - (1.96 * std / np.sqrt(nb_Simu)), 0)                 # Confidence interval
    b = proba + (1.96 * std / np.sqrt(nb_Simu))


    plt.plot(np.arange(1,nb_Simu+1),estimator,label = "Estimator of probability" )
    plt.axhline(a, color = 'g', label="Lower bound")
    plt.axhline(b, color = 'r', label = "Upper bound")
    plt.xlabel("Simulations")
    plt.ylabel("Estimator")
    plt.title('Convergence of the estimator of the probability of bad clustering with cin = 90 and cout = 72')
    plt.legend(loc='best')
    plt.show()
    print("Probability of bad clustering, with n =",n,", cin =",cin, ", cout =",cout, " is ", proba)
    print("Standard deviation : ", std)

    print("Confidence interval (95%) : [",a,",",b,"]")

def test_Monte_Carlo(set):
    """Tests Monte Carlo"""
    return(monteCarloApproach(nb_Simu = 10000, set_of_vertices = set))


def std_function_qin(N,nb_Simu,qout = 0.9):
    """Scatter plot of std depending on qin using model 1"""
    qin_space = np.linspace(0.1,0.9,N)
    std_space = []
    for qin in qin_space:
        simulation = importanceSamplingApproach(qin, qout, nb_Simu=nb_Simu, set_of_vertices=range(1), model=1)
        std = np.std(simulation)
        std_space.append(std)
    plt.scatter(qin_space,std_space)
    plt.xlabel("Value of the parameter qin")
    plt.ylabel("Standard deviation")
    plt.title("Standard deviation depending on qin")
    plt.show()

#std_function_qin(8,5000)

#print(test_Monte_Carlo(range(5)))

#print(test_importance_sampling(10000))

