import numpy as np
import matplotlib.pyplot as plt


ds = [
    ("PSO.12", np.load("data/free.pixmap.pso.12.81/results.npz")),
    ("PSO.12 harm3", np.load("data/free.pixmap.h3.pso.12.81/best.npz")),
    #("t12.5", np.load("data/free.pixmap.t.10.0/best.npz")),
    #("t12.6 (PSO)", np.load("data/free.pixmap.t.12.6/best.npz")),
    ]
for name, d in ds:
    profile = d["profile"]
    if len(profile.shape) == 2:
        iterations = profile[:, 0]
        profile = profile[:, 1]
    else:
        iterations = np.arange(len(profile))
    if np.any(profile<0):
        profile *= -1
    plt.plot(iterations, profile, marker=".",label=f"{name}")
plt.xlabel("Plane waves count in base lattice")
plt.ylabel("Metric [fraction of 1]")
plt.legend()
plt.savefig("opt_profiles.png")
