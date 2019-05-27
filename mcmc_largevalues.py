"""
   Purpose:

   Evaluate the probability of badly clustering a large set of vertices (n/4), with a graph ratio sufficiently low.
   The method used is splitting with MCMC simulation. We split the problem into K sub-problems, build Markov Chains for each sub-problem and
   evaluate a probability for each step with Monte-Carlo.

"""


import numpy as np
import spectral_clustering_1 as spc
import stochastic_block_model as sbm
import clustering_quality as qlt
import matplotlib.pyplot as plt


n = 100
cin = 90
cout = 85
k = 2

print(qlt.graph_ratio(cin, cout, n))
m = n//4  # Length of the set of vertices


def next_Markov_Graph(initial_graph, classes,p):
    """Returns next potential step in Markov Chain"""
    new_graph = sbm.simulate(n,cin,cout,k,classes)
    U = np.random.rand(n,n)
    change = U <= p  #Indicates which edges are to be modified
    return(new_graph*change + initial_graph*(1-change))

def MCMC_simulation(bound_split, p=0.6,M = 10000,set_of_vertices = range(m)):
    """Returns the matrix of simulated events in splitting method with MCMC: the matrix can be used for computing probabilities"""
    K = len(bound_split)-1
    simulation = np.zeros((M,K))
    classes = [0] * (n // 2) + [1] * (n - n // 2)
    initial_graph = sbm.simulate(n,cin,cout,k,classes) #Initial graph of the Markov Chain

    for i in range(K):   # Loop for each conditional probability
        a = bound_split[i]
        b = bound_split[i + 1]
        max = b  # max is the max value of phi found
        current_graph = initial_graph    #First step : we are sure that phi >=a, as phi is superior to the previous value of b
        clusters = qlt.reEvaluate(classes, spc.spectral_clustering(current_graph, k), n)
        simulation[0,i]= qlt.nb_of_misclustered(set_of_vertices,classes,clusters) >=b

        for j in range(1,M):
            new_graph = next_Markov_Graph(current_graph,classes,p)   #Potential next step
            new_clusters = qlt.reEvaluate(classes, spc.spectral_clustering(new_graph, k), n)
            phi = qlt.nb_of_misclustered(set_of_vertices,classes,new_clusters)
            current_graph = (phi>=a) * new_graph + (phi<a) * current_graph  # If there are enough misclustered vertices, we accept new_graph as the next step

            if (phi<a):
                simulation[j,i]=simulation[j-1,i]
            else :
                simulation[j, i] = (phi >= b)
                if phi >= max :          # If there are enough misclustered vertices, we can choose the graph as the initial step of the next Markov chain
                    initial_graph = current_graph
                    max = phi
    return (simulation)

def test_MCMC(p):
    """Tests splitting with MCMC, with specific thresholds and value of p"""
    bound_split = [0,15,20,21,22,23,24,25]  #Thresholds

    simulation = MCMC_simulation(bound_split, p=p)
    estimators = np.cumsum(simulation, axis = 0)/(np.arange(1,len(simulation[:,0])+1).reshape((-1,1)))

    intermediate_probas = estimators[-1,:]  # Array of conditional probabilities
    proba = np.prod(intermediate_probas)

    for i in range(len(estimators[0,:])):            #Plotting the estimators of conditional probabilities
            plt.figure(1)
            plt.plot(np.arange(1,len(simulation[:,0])+1), estimators[:,i], label='Estimator '+str(i+1))


    plt.legend(loc='best')
    plt.ylabel("Estimators of the conditional probabilities ")
    plt.xlabel("Simulations")
    plt.title('Convergence of the estimators of the conditional probabilities with cin = 90 and cout = 85')

    print('The thresholds are ', bound_split)
    for i in range(len(estimators[0,:])):
        print("Probability of having ", bound_split[i+1] ," badly clustered vertices knowing that we have ", bound_split[i], " of them is ", intermediate_probas[i])

    print("The final estimated probability is ", proba)

    plt.show()

    return(intermediate_probas)


#print(test_MCMC(0.6))


