from hybris.optim import ParticleSwarm, RuleSet, setup_opt_control
from functools import partial
import numpy as np
import logging
from copy import copy
import os
from keever.database import countinuous_variables_boundaries

from keever.database import (
    count_continuous_variables, 
    countinuous_variables_boundaries, 
    count_categorical_variables,
    categorical_variables_num_values)


def main(fevals, nagents, objective, doe, nd, workdir, seed, pso_weights, controller):
    ncat = count_categorical_variables(doe.variables_descr)
    opt = ParticleSwarm(nagents, [nd, ncat], max_fevals=fevals, 
        initial_weights=pso_weights)
    if controller != "none":
        print("Setting up a controller for PSO")
        crtl, opti_params = RuleSet.load(controller)
        setup_opt_control(opt, crtl)
    # Get the boundaries from the doe
    opt.vmin, opt.vmax = countinuous_variables_boundaries(doe.variables_descr)
    if ncat > 0:
        cats = categorical_variables_num_values(doe.variables_descr)
        opt.num_categories(cats)
    opt.reset(seed)
    
    best_fitness = -1
    best_design = None
    i = 0
    best_path = f"{workdir}/best.npz"
    while not opt.stop():
        X = opt.ask()
        # X.shape = (nagents, nd)
        fitness = np.asarray([objective(args={"design": x.copy()})["fitness"] for x in X])
        opt.tell(-fitness)
        iter_best = np.argmax(fitness)
        iter_fitness = fitness[iter_best]

        if iter_fitness > best_fitness:
            logging.info(f"New best!")
            best_fitness = copy(iter_fitness)
            best_design = X[iter_best].copy()
            np.savez(best_path, bd=best_design, bf=best_fitness, profile=opt.profile)
            figure_path = f"{workdir}/figs/cur_best_{i}.png"
            fitness = objective(args={"design": best_design.copy(), "figpath": figure_path})["fitness"]

        logging.info(f"{i=}, {iter_fitness=:0.3f}, {best_fitness=:0.3f}")
        i += 1

    fitness = objective(args={"design": best_design, "figpath": f"{workdir}/figs/best.png"})["fitness"]
    np.savez(best_path, bd=best_design, bf=best_fitness, profile=opt.profile)
    print(f"{best_design=}")
    return best_design, best_fitness, opt.profile


def __run__(fevals, nagents, fom, fom_method, doe, workdir, seed,pso_weights, controller):
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(workdir + "/figs", exist_ok=True)
    nd = doe.num_scalar_variables
    objective = partial(fom.action, name=fom_method)
    return main(fevals, nagents, objective, doe, nd, workdir, seed,pso_weights, controller)


def __requires__():
    return {"variables": ["fevals", "nagents", "fom", "fom_method", "doe", "workdir", "seed", "pso_weights", "controller"]}
