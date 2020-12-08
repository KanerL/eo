from math import fabs

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

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


a = -10
b = 10
k = 15
f = Levi13

error_or = 0




fit_eval = 0
max_epoch_count = 1000
max_fit_eval = 1000


epsilon = 10e-9
results = []
for _ in range(100):
    start_pop_x = np.array([random.randrange(a, b) for i in range(k)], dtype=float)
    start_pop_y = np.array([random.randrange(a, b) for i in range(k)], dtype=float)
    start_pop = np.array([start_pop_x, start_pop_y], dtype=float)
    fit_pop = f(*start_pop)
    best_res = min(fit_pop)
    all_best = best_res
    fit_avrg_prev = np.mean(fit_pop)
    epoch = 0
    print(f'Best fit {epoch} - {best_res}')
    errors = []
    while True:
        new_pop = []
        start_pop = list(start_pop.T)
        start_pop.sort(key=lambda x: f(*x))
        for i in range(k):
            start_pop.sort(key=lambda x: f(*x))
            xbest = start_pop[0]
            indexes = list(range(k))
            indexes.pop(i)
            xi = start_pop[i]
            j = random.choice(indexes)
            xj = start_pop[j]
            mutual_vector = (xi + xj) / 2
            bf1 = random.randrange(1, 3)
            bf2 = random.randrange(1, 3)
            xin = xi + random.random() * (xbest - mutual_vector * bf1)
            xjn = xj + random.random() * (xbest - mutual_vector * bf2)
            if f(*xin) < f(*xi) and f(*xjn) < f(*xj):
                start_pop[i] = xin
                start_pop[j] = xjn
            fit_eval += 2

            j = random.choice(indexes)
            xj = start_pop[j]
            xin = xi + (-1 + random.random() * 2) * (xbest - xj)
            if f(*xin) < f(*xi):
                start_pop[i] = xin
            fit_eval += 1

            j = random.choice(indexes)
            xj = start_pop[j]
            parasite = np.array([xi[0],xi[1]])
            parasite[random.randrange(0,2)] = random.randrange(a,b)
            if f(*parasite) < f(*xj):
                start_pop[j] = parasite
            fit_eval += 1
        start_pop.sort(key=lambda x: f(*x))
        start_pop = np.array(start_pop).T
        epoch_fit = f(*start_pop)
        max_epoch = epoch_fit[0]
        fit_avrg_epoch = np.mean(epoch_fit)
        # print(fit_avrg_epoch,fit_avrg_prev)

        if epoch >= max_epoch_count:
            break
        # if abs(best_res-max_epoch) < epsilon :
        #     break
        # if abs(fit_avrg_prev - fit_avrg_epoch) < epsilon:
        #     break
        fit_avrg_prev = fit_avrg_epoch
        best_res = max_epoch
        if all_best > max_epoch:
            all_best = max_epoch
        # print(f"{epoch} : Best res - {[start_pop[0][0], start_pop[1][0]]} with fit - {max_epoch}")
        epoch += 1
        errors.append(fabs(max_epoch-error_or))
    print(all_best)
    plt.plot(errors)
    plt.show()
    results.append(all_best)
print(np.mean(results))