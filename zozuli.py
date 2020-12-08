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
    return alpha*step
print(levy_walk())

a = -10
b = 10
k = 50
f = Levi13
# error_or = -39.166165*2
error_or = 0
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


epsilon = 10e-5
results = []
for _ in range(100):
    start_pop_x = np.array([random.randrange(a, b) for i in range(k)], dtype=float)
    start_pop_y = np.array([random.randrange(a, b) for i in range(k)], dtype=float)
    start_pop = np.array([start_pop_x, start_pop_y], dtype=float)

    fit_pop = f(*start_pop)
    best_res = min(fit_pop)
    epoch = 1
    print(f'Best fit {epoch} - {best_res}')
    fit_avrg_prev = np.mean(fit_pop)
    all_best = best_res
    errors = []
    while True:
        new_pop = []
        start_pop = list(start_pop.T)
        cucko_pos = start_pop[random.randrange(0, k)]
        cucko_pos = cucko_pos + levy_walk()
        cucko_pos[0] = min(cucko_pos[0],b)
        cucko_pos[0] = max(cucko_pos[0],-b)
        cucko_pos[1] = min(cucko_pos[1],b)
        cucko_pos[1] = max(cucko_pos[1],-b)

        cucko_fit = f(*cucko_pos)
        j = random.randrange(0, k)
        if cucko_fit < f(*start_pop[j]):
            start_pop[j] = cucko_pos
        start_pop.sort(key=lambda x: f(*x))
        for j in range(1,k):
            if random.random() < pa:
                start_pop[j] = start_pop[j] + levy_walk()
                start_pop[j][0] = min(start_pop[j][0], b)
                start_pop[j][0] = max(start_pop[j][0], -b)
                start_pop[j][1] = min(start_pop[j][1], b)
                start_pop[j][1] = max(start_pop[j][1], -b)

        start_pop.sort(key=lambda x: f(*x))
        start_pop = np.array(start_pop).T
        epoch_fit = f(*start_pop)
        max_epoch = epoch_fit[0]
        # if epoch % 100 == 0:
        # plt.figure(figsize=(10, 10))
        # ax = plt.axes(projection="3d")
        # ax.scatter(start_pop[0], start_pop[1], f(*start_pop), marker='o', c='r', s=100)
        # ax.plot_wireframe(X, Y, Z, rstride=2, cstride=1,
        #                       edgecolor=None)
        # ax.set(xlabel='x', ylabel='y', zlabel='f(x,y)',
        #            title=f'Epoch {epoch + 1}')
        # plt.show()
        # print(f"{epoch} : Best res - {[start_pop[0][0], start_pop[1][0]]} with fit - {max_epoch}")
        epoch += 1

        fit_avrg_epoch = np.mean(epoch_fit)
        # print(fit_avrg_epoch,fit_avrg_prev)

        if epoch >= max_epoch_count:
            break
        # if abs(best_res-max_epoch) < epsilon :
        #     break
        # if abs(fit_avrg_prev-fit_avrg_epoch) < epsilon :
        #     break
        fit_avrg_prev = fit_avrg_epoch
        best_res = max_epoch
        if all_best > max_epoch :
            all_best = max_epoch
        errors.append(fabs(max_epoch-error_or))
    print(f"Best res - {[start_pop[0][0], start_pop[1][0]]} with fit - {max_epoch}")
    results.append(all_best)
    plt.plot(errors)
    plt.show()
print(results)
print(np.mean(results))