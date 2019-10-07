# SNA_CommunityDetection

See the report for further detail about what is achieved by the code.

Here are the purposes of each file :

-clustering_quality: provides functions for evaluating the quality of clustering

-graph_drawing: draws a graph with suitable layout, and with specific colors for each class/cluster

-stochastic_block_model: Simulates a graph with the SBM, there is also a specific version for importance sampling

-spectral_clustering_1: Applies the spectral clustering algorithm on an adjacecncy matrix A

-test_misclustered_vertex: Computes probability of misclustering a small set of vertices using importance sampling

-mcmc_large_values: Computes probability of misclustering a large set of vertices using MCMC splitting

-test_ratio_hypothesis: Plots quality depending on graph ratio r

-test: Tests the algorithm of spectral clustering and displays graphs

