from numpy.random import rand

import numpy as np
from user.commons import freq2pix, default_harmonics

N = len(default_harmonics)

gratings = np.zeros((16, 128))
for i in range(16):
    theta, grating = freq2pix(rand(N), rand(N) *2*  np.pi)
    gratings[i] = grating

gratings = np.asarray(gratings)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.matshow(gratings, cmap="Blues", extent=[0, 1, 0, 1])
ax.set_xlabel("x [um]")
ax.set_ylabel("z [um]")
fig.savefig("figs/multigrating.png")


