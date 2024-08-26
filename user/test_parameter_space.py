import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from user.parameterization import fftlike, placeblocks
N = 5
N2 = N**2
harmonics = [0.5, 1, 1.5]
nh = len(harmonics)
configs = np.random.rand(N2, 12, 2*nh+1)
configs[:,:, nh:2*nh] /= 2

fig, axs = plt.subplots(N,N)
axs = axs.flatten()
for ax, X in zip(axs,configs):
    #gratings, depthss = fftlike(X, 2, 4, "free", harmonics=harmonics)
    gratings, depthss = placeblocks(X, 2, 4, "free", nh)
    bilayer_depth = np.sum(depthss, axis=1)
    gratings_picture = np.vstack((
        np.repeat( [g for g in gratings[0]], 
                np.round(100 * depthss[0] / np.sum(depthss)).astype(int), axis=0),
        np.repeat( [g for g in gratings[1]], 
                np.round(100 * depthss[1] / np.sum(depthss)).astype(int), axis=0),
    ))

    ax.matshow(gratings_picture, cmap="Blues")
    ax.axis("off")
plt.xlabel("Plane waves count in base lattice")
plt.ylabel("Metric [fraction of 1]")
plt.legend()
plt.savefig("test_parameterization.png")