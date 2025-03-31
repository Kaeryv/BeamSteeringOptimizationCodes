import numpy as np
from hybris.meta import optimize_self
import sys

benchmark_name = "harrison_2016"

# Target problem characteristics
# Target Use NE evals for ND problem using NA agents in swarm...
NE, NA, ND = int(sys.argv[2]), 40, 120


mNA = 40
mNE = 12000
dbname = "./warmup_120.npy"
opti_args = {
                "max_fevals": NE,
                "num_variables": [ND,0], 
                "num_agents": NA ,
                "initial_weights": [0.7298, 1.49618, 1.49618, 0.0, -16, 0.6, 1.0 ],
            }#[0.679,1.49618,1.49618,0.0,-16,0.6,1.0] on cluster
seed = int(sys.argv[1])
metaprofile, best_configuration, optidata, db = optimize_self(
    sys.argv[4], seed=seed, optimize_membership=True, db=dbname, return_all=True, 
    profiler_args={
        "benchmark": benchmark_name,
        "optimizer_args_update": opti_args,
        "max_workers": 16
    }, metaopt_args = { # Arguments of the optimizer of the optimizer
        "type": "pso",
        "initial_weights": [0.7298, 1.49618, 1.49618, 0.0, -16, 0.6, 0.5 ],
        "max_fevals": mNE,
        "num_agents": mNA
    })
    
np.save(dbname, db)
best_configuration.save(sys.argv[3], more=opti_args)
