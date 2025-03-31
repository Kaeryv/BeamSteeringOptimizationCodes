import cma
from functools import partial
import numpy as np
import logging
from copy import copy
import os
from keever.database import countinuous_variables_boundaries

def main(fevals, nagents, objective, doe, nd, workdir):
    vmin, vmax =  countinuous_variables_boundaries(doe.variables_descr)
    center = (vmax - vmin) / 2 + vmin
    es = cma.CMAEvolutionStrategy(center, 0.5, {
        'BoundaryHandler': cma.s.ch.BoundTransform,
        'bounds': [vmin, vmax],
        'maxiter': fevals//nagents,
        'popsize': nagents
        #'tolfun': 0.0,
        #'tolx': 0.0,
        #'tolstagnation': np.inf
    }, )#{'bounds': [0, 1]}
    best_fitness = 0
    best_design = None
    profile = list()
    i = 0

    best_path = f"{workdir}/best.npz"
    while not es.stop():
        X = es.ask()
        fitness = np.asarray([objective(args={"design": x.copy()})["fitness"] for x in X])
        es.tell(X, [-f for f in fitness])

        iter_best = np.argmax(fitness)
        iter_fitness = fitness[iter_best]

        if iter_fitness > best_fitness:
            logging.info("New best!")
            best_fitness = copy(iter_fitness)
            best_design = X[iter_best].copy()
            np.savez(best_path, bd=best_design, bf=best_fitness, profile=profile)

            fitness = objective(args={"design": best_design.copy(), "figpath": f"{workdir}/figs/cur_best_{i}.png"})
            profile.append((i, best_fitness))
        
        logging.info(f"{i=}, {iter_fitness=:0.3f}, {best_fitness=:0.3f}")
        i += 1


    np.savez(best_path, bd=best_design, bf=best_fitness, profile=profile)

    return best_design, best_fitness, np.asarray(profile)

def __run__(fevals, nagents, fom, fom_method, doe, workdir, **kwargs):
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(workdir + "/figs", exist_ok=True)
    nd = doe.num_scalar_variables
    objective = partial(fom.action, name=fom_method)
    return main(fevals, nagents, objective, doe, nd, workdir)


def __requires__():
    return {"variables": ["fevals", "nagents", "fom", "fom_method", "doe", "workdir"]}
