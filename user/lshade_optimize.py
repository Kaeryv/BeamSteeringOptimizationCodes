from functools import partial
import numpy as np
import logging
from copy import copy
import os
from shady.lshade import optimize_lshade

def main(fevals, nagents, objective, doe, nd, workdir, seed):
    global i, best_fitness, profile
    np.random.seed(seed)
    bounds = np.zeros((2, nd))
    bounds[0] = 0.0
    bounds[1] = 1.0

    i = 0
    profile = list()
    best_fitness = 0
    best_path = f"{workdir}/best.npz"
    def objective_function(X):
        global i, best_fitness, profile
        fitness = np.asarray([objective(args={"design": x.copy()}) for x in X])

        iter_best = np.argmax(fitness)
        iter_fitness = fitness[iter_best]

        if iter_fitness > best_fitness:
            logging.info(f"New best!")
            best_fitness = copy(iter_fitness)
            profile.append((i, best_fitness))
            best_design = X[iter_best].copy()
            np.savez(best_path, bd=best_design, bf=best_fitness, profile=profile)
            figure_path = f"{workdir}/figs/cur_best_{i}.png"
            objective(args={"design": best_design.copy(), "figpath": figure_path})
        logging.info(f"{i=}, {iter_fitness=:0.3f}, {best_fitness=:0.3f}")
        i += 1
        return - fitness

    profile, best_design = optimize_lshade(objective_function, bounds, max_nfes=fevals, return_solution=True)
    best_fitness = profile[-1]
    fitness = objective(args={"design": best_design, "figpath": f"{workdir}/figs/best.png"})
    np.savez(best_path, bd=best_design, bf=best_fitness, profile=profile)

    return best_design, best_fitness, profile


def __run__(fevals, nagents, fom, fom_method, doe, workdir, seed):
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(workdir + "/figs", exist_ok=True)
    nd = doe.num_scalar_variables
    objective = partial(fom.action, name=fom_method)
    return main(fevals, nagents, objective, doe, nd, workdir, seed)


def __requires__():
    return {"variables": ["fevals", "nagents", "fom", "fom_method", "doe", "workdir", "seed"]}
