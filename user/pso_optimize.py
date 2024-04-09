from hybris.optim import ParticleSwarm
from functools import partial
import numpy as np
import logging
from copy import copy


def main(fevals, nagents, objective, doe, nd):
    opt = ParticleSwarm(nagents, [nd, 0], max_fevals=fevals)
    opt.vmin = 0
    opt.vmax = 1
    opt.reset(42)

    best_fitness = 0
    best_design = None

    while not opt.stop():
        X = opt.ask()
        # X.shape = (nagents, nd)
        fitness = np.asarray([objective(args={"design": x}) for x in X])
        opt.tell(-fitness)
        iter_best = np.argmax(fitness)
        iter_fitness = fitness[iter_best]

        if iter_fitness > best_fitness:
            logging.info(f"New best!")
            best_fitness = copy(iter_fitness)
            best_design = X[iter_best].copy()

        logging.info(f"{iter_fitness=}, {best_fitness=}")

    fitness = objective(args={"design": best_design, "figpath": "figs/best.png"})
    fitness = objective(args={"design": best_design, "figpath": "figs/best2.png"})
    np.savez("best.npz", bd=best_design, bf=best_fitness)

    return best_design, best_fitness, opt.profile


def __run__(fevals, nagents, fom, fom_method, doe):
    nd = doe.num_scalar_variables
    objective = partial(fom.action, name=fom_method)
    return main(fevals, nagents, objective, doe, nd)


def __requires__():
    return {"variables": ["fevals", "nagents", "fom", "fom_method", "doe"]}
