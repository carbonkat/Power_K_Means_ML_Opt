# Power_K_Means_ML_Opt
A repository for modified code from the paper "Power k-Means Clustering" for RPI ML and Optimization Spring 2023

# To Use
This code runs using Python3 with additional support from the machine learning libraries sklearn and PyTorch.

Run experiments using the command python run_experiments.py. Additional experiments can be run by modifying variables within the run_experiments.py file.

Synthetic datasets are seeded so generation is consistent across multiple runs. Default dataset settings use a binomial distribution of 99 data points across 3 clusters. Experiments are run across 250 trials for each Power k-Means initial value $s_0$, with metric results (ARI, NMI, VI) being averaged to obtain final performance values.

Default experiment settings use $s_0$ initializations of -18.0, -9.0, -3.0, -2.0, and -1.0. Default dimensions are 2, 10, 50, 100, 200, 500, 800, 1000, 1500, 1800, 2000, and 5000.

Optional support is added for cluster plotting in 2-D and result graphing across all three metrics. Metrics have the option of being plotted using either individual clusters for each Power k-Means initialization or color-coded such that vanilla k-Means is plotted in blue and all versions of Power k-Means are plotted in red for better visualization. Default settings use the two-color plotting scheme, but this can be changed by switching the colors parameter in make_plots to 0.

## For real datasets, the High dimensional data must be downloaded from this site https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq  as it is too large for github, but the label set must be from this repository ("gene_labels.csv"), as the labels were pre processed.

# Notes
The only modifications that should be made are in the "main" function.
CSV files containing numerical results and their respective plots are located in the Results folder.
