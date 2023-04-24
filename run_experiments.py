import csv
import os
import time
from datetime import datetime
from uuid import uuid4

import numpy as np
import torch

from utils import *
from euclidean_k_means import kmeans, power_kmeans

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def setup_experiment(init_params):
    init_params_name_copy = init_params['data_params'].copy()
    init_params_name_copy['s_0'] = init_params['s_0']
    del init_params_name_copy['center_box']
    del init_params_name_copy['center_coordinates']
    test = "n_samples_" + str(init_params_name_copy['n_samples']) + "_n_features_" + str(init_params_name_copy['n_features']) + "_centers_" + str(init_params_name_copy['centers']) + "_data_dist_" + str(init_params_name_copy['data_dist']) + "_desired_variance_" + str(init_params_name_copy['desired_variance']) + "_s_0_" + str(init_params_name_copy['s_0'])
    init_params_name_copy = test
    data_lambda = lambda i: generate_points(random_state=i, **init_params['data_params'])

    return data_lambda


def run_experiment(init_params):
    start_exp = time.time()
    data_func = setup_experiment(init_params)

    k = init_params['data_params']['centers']

    iters_og = []
    times_og = []
    VIs_og = []
    ARIs_og = []
    NMIs_og = []

    iters_power = []
    times_power = []
    VIs_power = []
    ARIs_power = []
    NMIs_power = []

    i = 0
    cnt = 0

    start_trial = time.time()
    sample_dataset = []
    #run n trials for each experiment
    while cnt < init_params['n_trials']:
        X, classes_true, centers_true = data_func(i)
        X = torch.tensor(X)
        X = X.to(torch.float64)


        centers_init = initcenters(X, k, random_state=i) #want to keep consistent center initialization

        #lloyd's, power k-means, bregman iterative, bregman closed form
        classes_og, centers_og, iter_og, time_og = kmeans(X, k, centers_init)
        classes_power, centers_power, s_final_power, iter_power, time_power = power_kmeans(X, init_params['s_0'], k, centers_init, classes_true)
        
        if i == 0:
            sample_dataset = [X,classes_true,classes_og,classes_power]
        i += 1

        iters_og += [iter_og]
        iters_power += [iter_power]

        times_og += [time_og]
        times_power += [time_power]

        VIs_og += [VI(k, classes_true, k, classes_og)]
        VIs_power += [VI(k, classes_true, k, classes_power)]

        ARIs_og += [adjusted_rand_score(classes_true, classes_og)]
        ARIs_power += [adjusted_rand_score(classes_true, classes_power)]

        NMIs_og += [normalized_mutual_info_score(classes_true, classes_og)]
        NMIs_power += [normalized_mutual_info_score(classes_true, classes_power)]

        cnt += 1

        start_trial = time.time()

    iters_og = np.array(iters_og)
    times_og = np.array(times_og)
    VIs_og = np.array(VIs_og)
    ARIs_og = np.array(ARIs_og)
    NMIs_og = np.array(NMIs_og)

    iters_power = np.array(iters_power)
    times_power = np.array(times_power)
    VIs_power = np.array(VIs_power)
    ARIs_power = np.array(ARIs_power)
    NMIs_power = np.array(NMIs_power)

    sqrt_n = np.sqrt(init_params['n_trials'])

    #show_dataset(sample_dataset)
    print("Experiment Done. Dimension: ", init_params['data_params']['n_features'], "s_0: ", init_params['s_0'], ", Time Elapsed (sec): ", time.time() - start_exp)
    return [init_params['data_params']['n_features'],np.mean(VIs_og), np.mean(ARIs_og), np.mean(NMIs_og), np.mean(VIs_power), np.mean(ARIs_power), np.mean(NMIs_power)]

def run_experiment_group(init_params, s_0s, exp_dir='exp_dir', exp_id=None):
    df = 0
    for s_0 in s_0s:
        init_params['s_0'] = s_0
        print("running experiment, s_0: ", init_params['s_0'])
        df = run_experiment(init_params)

    return df

def show_dataset(info):
    X = info[0]
    true_classes = np.array(info[1])
    k_means_classes = np.array(info[2])
    power_classes = np.array(info[3])

    plt_data, data_ax = plt.subplots(figsize=(10,10))
    data_ax.set_title('True Clusters in 2 Dimensions')
    data_ax.set(xlabel='X axis',ylabel='Y axis')

    plt_og, og_ax = plt.subplots(figsize=(10,10))
    og_ax.set_title('k-Means Clusters in 2 Dimensions')
    og_ax.set(xlabel='X axis',ylabel='Y axis')

    plt_pwer, pwer_ax = plt.subplots(figsize=(10,10))
    pwer_ax.set_title('Power k-Means Clusters in 2 Dimensions')
    pwer_ax.set(xlabel='X axis',ylabel='Y axis')
    colors = ['blue', 'red', 'purple']
    for cl in np.unique(true_classes):
        l = 'Cluster ' + str(cl)
        cluster_mems = np.where(true_classes==cl)
        points = X[cluster_mems]

        cluster_mems_k = np.where(k_means_classes==cl)
        points_k = X[cluster_mems_k]

        cluster_mems_pwer = np.where(power_classes==cl)
        points_pwer = X[cluster_mems_pwer]

        data_ax.scatter(points[:,0], points[:,1], label=l, color=colors[cl])
        og_ax.scatter(points_k[:,0], points_k[:,1], label=l, color=colors[cl])
        pwer_ax.scatter(points_pwer[:,0], points_pwer[:,1], label=l,color=colors[cl])

    plt_data.legend()
    plt_og.legend()
    plt_pwer.legend()
    plt_data.savefig('true_clusters.png')
    plt_og.savefig('k_means_clusters.png')
    plt_pwer.savefig('power_means_clusters.png')
    

def make_plots(dfs, colors):
    '''
    dfs: list of dataframes for each k-Means experiment
    colors: activates/deactivates color scheme in plotting:
        1: two-color scheme
        0: all color scheme
    '''
    x = dfs[0][0]['Features']
    plt_nmi, nmi_ax = plt.subplots(figsize=(10,10))
    nmi_ax.set_title('Mean NMI')
    nmi_ax.set(xlabel='Number of Features',ylabel='Mean NMI')

    plt_ari, ari_ax = plt.subplots(figsize=(10,10))
    ari_ax.set_title('Mean ARI')
    ari_ax.set(xlabel='Number of Features',ylabel='Mean ARI')

    plt_vi, vi_ax = plt.subplots(figsize=(10,10))
    vi_ax.set_title('Mean VI')
    vi_ax.set(xlabel='Number of Features',ylabel='Mean VI')

    nmi_ax.plot(x, dfs[0][0]['NMI Lloyd'], label='k-Means', color='blue')
    ari_ax.plot(x, dfs[0][0]['ARI Lloyd'], label='k-Means', color='blue')
    vi_ax.plot(x, dfs[0][0]['VI Lloyd'], label='k-Means', color='blue')
    for tup in dfs:
        df = tup[0]
        s_0 = tup[1]
        c = ''
        if colors == 1:
            c = 'red'

        label_2 = 'Power k-Means_' + s_0
        
        nmi_ax.plot(x, df['NMI Power'], label=label_2, color=c)

        
        ari_ax.plot(x, df['ARI Power'], label=label_2, color=c)

        
        vi_ax.plot(x, df['VI Power'], label=label_2, color=c)

        table_name = s_0 + '_table.csv'
        df = df.round(3)
        df.to_csv('Results/'+table_name)

    plt_nmi.legend()
    plt_ari.legend()
    plt_vi.legend()

    nmi_name = 'nmi'
    ari_name = 'ari'
    vi_name = 'vi'
    if colors == 1:
        nmi_name = nmi_name + '_2color.png'
        ari_name = ari_name + '_2color.png'
        vi_name = vi_name + '_2color.png'
    plt_nmi.savefig('Results/'+nmi_name)
    plt_ari.savefig('Results/'+ari_name)
    plt_vi.savefig('Results/'+vi_name)


if __name__ == "__main__":
    k = 3
    d = 2
    
    init_params_gaussian = {
        'n_trials': 250,
        'bregman_dist': 'gaussian',
        'data_params': {
            'n_samples': 99, 
            'n_features': d,
            'center_box': (1, 40),
            'center_coordinates': np.array([[10]*d, [20]*d, [40]*d]),
            'centers': k,
            'data_dist': 'gaussian',
            'desired_variance': 16.0,
        },
        'convergence_threshold': 5
    }

    init_params_binomial = {
            'n_trials': 250,
            'bregman_dist': 'multinomial',
            'data_params': {
                'n_samples': 99,
                'n_features': d,
                'center_box': (1, 40),
                'center_coordinates': np.array([[10]*d, [20]*d, [40]*d]),
                'centers': k,
                'data_dist': 'multinomial',
                'desired_variance': None, #isn't used with multinomial
            },
            'convergence_threshold': 10
    }

    init_params_poisson = {
            'n_trials': 250,
            'bregman_dist': 'poisson',
            'data_params': {
                'n_samples': 99, 
                'n_features': d,
                'center_box': (1, 40),
                'center_coordinates': np.array([[10]*d, [20]*d, [40]*d]),
                'centers': k,
                'data_dist': 'poisson',
                'desired_variance': None, #isn't used with poisson
            },
            'convergence_threshold': 10
        }

    init_params_gamma = {
            'n_trials': 250,
            'bregman_dist': 'gamma',
            'data_params': {
                'n_samples': 99, 
                'n_features': 2,
                'center_box': (1, 40), #they did 10,20,40 in robust bregman clustering: https://arxiv.org/pdf/1812.04356.pdf
                'centers': 3,
                'center_coordinates': np.array([[10,10], [20,20], [40,40]]),
                'data_dist': 'gamma',
                'desired_variance': None, #isn't used with gamma
                'shape': 3.0,
            },
            'convergence_threshold': 10
    }
    dims = [2,10,50,100,200, 500, 800, 1000, 1500, 1800, 2000, 5000]
    s_0_list = [-18.0, -9.0, -3.0, -2.0, -1.0]
    dfs = []

    for s_0 in s_0_list:
        df = pd.DataFrame(columns=["Features","VI Lloyd", "ARI Lloyd", "NMI Lloyd", "VI Power", "ARI Power", "NMI Power"])
        name = str(s_0)
        name = name.replace('-','neg')
        for d in dims:
            init_params_binomial['data_params']['n_features'] = d
            init_params_binomial['data_params']['center_coordinates'] = np.array([[10]*d, [20]*d, [40]*d])
            result = run_experiment_group(init_params_binomial, s_0s=[s_0])
            df.loc[len(df)] = result
        dfs.append((df,name))
        print(df.head())

    make_plots(dfs, 1)