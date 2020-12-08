from math import fabs

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
from matplotlib import cm
from scipy.stats import uniform, levy

"""
11, 20, 10
Хіммельблау

Стибінсього-Танга

Леві № 13



"""


def Himmelblau(x, y):
    return np.power(np.power(x, 2) + y - 11, 2) + np.power(x + np.power(y, 2) - 7, 2)


def Styb_Tang(x, y):
    return ((np.power(x, 4) - 16 * np.power(x, 2) + 5 * x) + (np.power(y, 4) - 16 * np.power(y, 2) + 5 * y)) / 2


def Levi13(x, y):
    return np.power(np.sin(3 * np.pi * x), 2) + np.power(x - 1, 2) * (
            1 + np.power(np.sin(3 * np.pi * y), 2)) + np.power(y - 1, 2) * (
                   1 + np.power(np.sin(2 * np.pi * y), 2))


Lambda = 1.5
alpha = 1
pa = 0.1


def levy_walk():
    sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / np.math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=2)
    v = np.random.normal(0, sigma2, size=2)
    step = u / np.power(np.fabs(v), 1 / Lambda)
    return alpha * step


print(levy_walk())

a = -5
b = 5
k = 2
f = Himmelblau

# error_or = 0
error_or = -39.166165*2

tabu_list_len = 50
tabu_node_eps = 10e-3
neighbor_count = 30
neighbor_prob = 0.9
neighbor_search_range = 0.4
max_epoch_count = 1000

x = np.linspace(a, b)
y = np.linspace(a, b)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# plt.figure(figsize=(10, 10))
# ax = plt.axes(projection="3d")
# ax.scatter(start_pop[0], start_pop[1], fit_pop, marker='o', c='r', s=100)
# ax.plot_wireframe(X, Y, Z, rstride=2, cstride=1,
#                   edgecolor=None)
# ax.set(xlabel='x', ylabel='y', zlabel='f(x,y)',
#        title='Epoch 0')
# plt.show()
def generate_neighbor(x, y):
    x = np.linspace(x-neighbor_search_range,x+neighbor_search_range)
    y = np.linspace(y-neighbor_search_range,y+neighbor_search_range)
    x,y = np.meshgrid(x,y)
    z = f(x ,y)
    minzx= np.argmin(z,axis=0)[0]
    minzy= np.argmin(z,axis=1)[0]
    return np.array([x[minzx][minzy],y[minzy][minzy]])

def generate_neighbor2(x, y):
    neighbors_x = []
    neighbors_y = []
    for _ in range(neighbor_count):
        if random.random() <= neighbor_prob:
            neighbors_x.append(x+random.uniform(-neighbor_search_range,neighbor_search_range))
            neighbors_y.append(y+random.uniform(-neighbor_search_range,neighbor_search_range))
    neighbors_x = np.array(neighbors_x)
    neighbors_y = np.array(neighbors_y)
    fit_neigh_min_index = np.argmin(f(neighbors_x,neighbors_y))
    return np.array([neighbors_x[fit_neigh_min_index],neighbors_y[fit_neigh_min_index]])

def in_tabu_list(point, tabu_list):
    for i in range(len(tabu_list)-1, -1, -1):
        dist = np.sqrt(np.sum((point - tabu_list[i]) ** 2, axis=0))
        if dist <= tabu_node_eps:
            return True
    return False


epsilon = 10e-5
results = []
for _ in range(100):
    start_pop_x = np.array([random.randrange(a, b) for i in range(k)], dtype=float)
    start_pop_y = np.array([random.randrange(a, b) for i in range(k)], dtype=float)
    start_pop = np.array([start_pop_x, start_pop_y], dtype=float)
    tabu_lists = [[] for _ in range(k)]
    fit_pop = f(*start_pop)
    best_res = min(fit_pop)
    epoch = 1
    # print(f'Best fit {epoch} - {best_res}')
    fit_avrg_prev = np.mean(fit_pop)
    all_best = best_res
    all_best_point = [0,0]
    start_pop = start_pop.T
    errors = []
    while True:
        for i in range(k):
            f_curr = f(*start_pop[i])
            neig_point = generate_neighbor(*start_pop[i])
            while in_tabu_list(neig_point,tabu_lists[i]):
                neig_point = generate_neighbor2(*start_pop[i])
            tabu_lists[i].append(start_pop[i])
            if f(*neig_point) < all_best:
                all_best = f(*neig_point)
                all_best_point = neig_point
            start_pop[i] = neig_point
            if len(tabu_lists[i]) > tabu_list_len :
                tabu_lists[i].pop(0)
        start_pop = list(start_pop)
        start_pop.sort(key=lambda x: f(*x))
        start_pop = np.array(start_pop).T
        epoch_fit = f(*start_pop)
        max_epoch = epoch_fit[0]
        # print(f"Best res epoch# {epoch}- {all_best_point} with fit - {all_best}")
        epoch += 1
        start_pop = start_pop.T
        fit_avrg_epoch = np.mean(epoch_fit)
        # if epoch > max_epoch_count :
        #     break
        if abs(all_best-max_epoch) < epsilon :
            break
        # if abs(fit_avrg_prev - fit_avrg_epoch) < epsilon:
        #     break

        fit_avrg_prev = fit_avrg_epoch
        best_res = max_epoch
        errors.append(fabs(all_best - error_or))
    results.append(all_best)
    # plt.plot(errors)
    # plt.show()
    print(f"Best res epoch# {_}- {all_best_point} with fit - {all_best}")
print(results)
print(np.mean(results))
