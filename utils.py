import time
import os
import sys
import cloudpickle
from matplotlib import pyplot as plt

from math import sqrt
import numpy as np
import torch

def generate_points(random_state, **kwargs):
    np.random.seed(random_state)

    n = kwargs['n_samples']
    d = kwargs['n_features']
    k = kwargs['centers']
    (lo, hi) = kwargs['center_box']

    assert n % k == 0 #ensure that each cluster will have same number of points
 
    if 'center_coordinates' in kwargs.keys():
        assert kwargs['center_coordinates'].shape == (k,d)
        centers = kwargs['center_coordinates']
    else:
        centers = np.random.rand(k, d) * (hi - lo) + lo #generate centers uniformly from (lo, hi)^d

    #generate points
    X = np.zeros([n, d], dtype=np.float64)
    y_true = np.zeros([n], dtype=np.int64)
    cluster_size = n // k

    for i in range(k):
        if kwargs['data_dist'] == 'gamma':
            X[i*cluster_size : (i+1)*cluster_size, :] = random_sample(dist_name=kwargs['data_dist'], center=centers[i], desired_variance=kwargs['desired_variance'], n=cluster_size, d=d, shape=kwargs['shape'])
        else:
            X[i*cluster_size : (i+1)*cluster_size, :] = random_sample(dist_name=kwargs['data_dist'], center=centers[i], desired_variance=kwargs['desired_variance'], n=cluster_size, d=d)

        y_true[i*cluster_size : (i+1)*cluster_size] = i

    X, y_true = parallel_shuffle(X, y_true) #randomly shuffle points and labels
    return X, y_true, centers


def random_sample(dist_name, center, desired_variance, n, d, shape=None):
    dist_dict = {
        'gaussian': lambda center: np.random.normal(loc=center, scale=sqrt(desired_variance), size=(n,d)),
        'multinomial': lambda center: np.random.binomial(n=50, p=center[0]/50, size=(n,d)), 
        'exponential': lambda center: np.random.exponential(scale=center, size=(n,d)),
        'poisson': lambda center: np.random.poisson(lam=center, size=(n,d)),
        'gamma': lambda center, shape: np.random.gamma(shape=shape, scale=center/shape, size=(n,d))
    }

    sample_func = dist_dict[dist_name]

    if shape:
        return sample_func(center, shape).astype(np.float64)
    else:
        return sample_func(center).astype(np.float64)



####################################################################################################################



def seed2center(init_seeds, X):
    return X[init_seeds, :]


def seed2class(init_seeds, X):
    return get_classes(seed2center(init_seeds, X))


'''
performs class assignments based on center locations

inputs: X (n rows of m-dimensional points),
        centers (k rows of m-dimensional points)
output: classes (n-dimensional vector)
'''
def get_classes(X, centers, phi):
    bregman_dist = pairwise_bregman(X, centers, phi) #n x k distance matrix
    classes = torch.argmin(bregman_dist, axis=1) #n x 1 - new classifications for each point
    return classes


'''
performs k-means ++ class initializations

inputs: X (n rows of m-dimensional points),
        k (scalar)
output: classes (n-dimensional vector)
'''
def initclass(X, k, phi):
    centers = initcenters(X, k, phi)
    classes = get_classes(X, centers, phi)
    return classes


'''
performs k-means ++ class initializations

inputs: X (n rows of m-dimensional points),
        k (scalar)
output: centers (k rows of m-dimensional points)
'''
def initcenters(X, k, random_state, phi=None):
    np.random.seed(random_state)
    return torch.tensor(np.random.randint(low=torch.min(X), high=torch.max(X), size=(k,X.shape[1])))


'''
NOTE: not in use at the moment

performs k-means ++ class initializations

inputs: X (n rows of m-dimensional points),
        k (scalar)
output: init_seeds (k-dimensional vector specifying indices of the k centroids in X)
'''
def initseeds(X, k, phi, **kwargs):
    n_points, m_features = X.size()

    init_seeds = torch.zeros([k], dtype=torch.int)
    classes = torch.zeros([n_points], dtype=torch.int)

    #seed the generators to ensure consistent cluster initialization
    gen = torch.Generator()
    gen.manual_seed(kwargs['init_seed'])

    p = torch.randint(n_points, size=(1,), generator=gen)
    init_seeds[0] = p

    if k > 1:
        min_costs = torch.squeeze(pairwise_bregman(X, X[p,:], phi, **kwargs)) #n x 1 distance matrix
        min_costs[p] = 0

        #Pick remaining centroids with probability proportional to mincost
        tmp_costs = torch.zeros([n_points], dtype=torch.float)
        for j in range(1, k):
            p = torch.multinomial(min_costs, 1, generator=gen)
            init_seeds[j] = p
            tmp_costs = torch.squeeze(pairwise_bregman(X, X[p,:], phi, **kwargs))
            min_costs = torch.min(min_costs, tmp_costs) #only update costs if new costs lower than old one
            min_costs[p] = 0

    init_seeds = init_seeds.long()
    return init_seeds



####################################################################################################################



def dist_to_phi(dist):
    dist_to_phi_dict = {
        'gaussian': 'euclidean',
        'multinomial': 'kl_div',
        'exponential': 'itakura_saito',
        'poisson': 'relative_entropy',
        'gamma': 'gamma'
    }
    return dist_to_phi_dict[dist]


'''
this function is structured weirdly: first 2 entries (phi, gradient of phi) can handle n x m theta matrix
last entry, only designed to work in iterative bregman update function, only works with 1 x m theta matrix and thus returns an m x m hessian
'''
def get_phi(name):
    phi_dict = {
        'euclidean': [lambda theta: torch.sum(theta**2, axis=1), lambda theta: 2*theta, lambda theta: 2*torch.eye(theta.size()[1], dtype=torch.float64)],
        'kl_div': [lambda theta: torch.sum(theta * torch.log(theta), axis=1), lambda theta: torch.log(theta) + 1, lambda theta: torch.eye(theta.size()[1], dtype=torch.float64) * 1/theta],
        'itakura_saito': [lambda theta: torch.sum(-torch.log(theta) - 1, axis=1), lambda theta: -1/theta, lambda theta: torch.eye(theta.size()[1]) / (theta**2)],
        'relative_entropy': [lambda theta: torch.sum(theta * torch.log(theta) - theta, axis=1), lambda theta: torch.log(theta), lambda theta: torch.eye(theta.size()[1]) / theta],
        'gamma': [lambda theta, k: torch.sum(-k + k * torch.log(k/theta), axis=1), lambda theta, k: -k/theta, lambda theta, k: k * torch.eye(theta.size()[1]) / (theta**2)]
    }
    return phi_dict[name]


#x, theta are both k-dimensional
def bregman_divergence(phi_list, x, theta):
    phi = phi_list[0]
    gradient = phi_list[1]

    bregman_div = phi(x) - phi(theta) - torch.dot(gradient(theta), x-theta)
    return bregman_div


#X is n x m, y is k x m, output is n x k containing all the pairwise bregman divergences
def pairwise_bregman(X, Y, phi_list, shape=None):
    phi = phi_list[0]
    gradient = phi_list[1]

    if shape:
        phi_X = phi(X, shape)[:, np.newaxis]
        phi_Y = phi(Y, shape)[np.newaxis, :]
    else:
        phi_X = phi(X)[:, np.newaxis]
        phi_Y = phi(Y)[np.newaxis, :]

    X = X[:, np.newaxis]
    Y = Y[np.newaxis, :]


    if shape:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y, shape), axis=-1)
    else:
        pairwise_distances = phi_X - phi_Y - torch.sum((X - Y) * gradient(Y), axis=-1)

    return torch.clamp(pairwise_distances, min=1e-12, max=1e6)


'''
Input: x is a Nxd matrix
        y is an optional Mxd matrix
Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        if y is not given then use 'y=x'.
i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
'''
def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    pairwise_distances = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(pairwise_distances, min=1e-12, max=1e6)


'''
computes power mean

inputs: m (k-dimensional vector), s (scalar)
output: s-mean of m (scalar)
'''
def powmean(m, s):
    res = torch.pow(torch.mean(torch.pow(m, s)), 1/s)

    if res > 0:
        return res

    #to prevent underflow, return min of original if the mean is rounded to 0
    return torch.min(m)



####################################################################################################################



#returns adjusted rand indices based on c1, c2, two n-dimensional vectors listing class membership
def ARI(c1, c2):
    return adjusted_rand_score(c1.numpy(), c2.numpy()) 


from math import log
def VI(k1, a1, k2, a2):
    X = [[] for i in range(k1)]
    Y = [[] for i in range(k2)]

    for i, el in enumerate(a1):
        X[el] += [i]

    for i, el in enumerate(a2):
        Y[el] += [i]


    n = float(sum([len(x) for x in X]))
    sigma = 0.0

    for x in X: #X_i
        p = len(x) / n

        for y in Y: #Y_j
            q = len(y) / n
            r = len(set(x) & set(y)) / n

            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    return abs(sigma)



####################################################################################################################



def get_object_size(obj):
    return sys.getsizeof(cloudpickle.dumps(obj))


def parallel_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



####################################################################################################################
def visualize_lineplot(vals, times, line_names=None, x_axis_name=None, y_axis_name=None, colors=None, dpi=500, width_height_factors=[9, 6], save_path_list=None):
    assert (1 <= len(vals.shape) == len(times.shape) <= 2)
    assert (len(line_names) == 1)

    plt.figure(figsize=(width_height_factors[0], width_height_factors[1]), dpi=dpi)
    plt.cla()

    max_x_val = 0
    min_y_val = np.min(vals)
    max_y_val = np.max(vals)
    y_range = max_y_val - min_y_val

    colors = ['r', 'g', 'b', 'k', 'c', 'm']

    plt.plot(times, vals, color=colors[0], label=line_names[0], markersize=10)

    if x_axis_name is not None: plt.xlabel(x_axis_name, fontsize=16)
    if y_axis_name is not None: plt.ylabel(y_axis_name, fontsize=16)

    plt.grid()
    plt.legend(frameon=True)

    if (min_y_val-0.1*y_range) != (max_y_val+0.1*y_range):
        plt.ylim((min_y_val-0.1*y_range, max_y_val+0.1*y_range))
    if 0 != (np.max(times)):
        plt.xlim((0, np.max(times)))

    full_save_path_list = []
    if save_path_list is None:
        plt.show()

    else:
        for path in save_path_list:
            path_dir = path[:-((path[::-1]).find('/'))]
            if not os.path.exists(path_dir): os.makedirs(path_dir)

            full_path = path[:-4] +'_LinePlot.png' 
            plt.savefig(full_path, dpi=dpi, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
            full_save_path_list.append(full_path)

    plt.close()
    return full_save_path_list
