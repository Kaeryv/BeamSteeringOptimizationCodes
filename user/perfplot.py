import numpy as np
import matplotlib.pyplot as plt
params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'font.family': 'serif',
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   'figure.figsize': [5.5, 4.5]
   }
plt.rcParams.update(params)
layers = np.array([8, 10, 12,14,16, 18,20])
layers_perfs = [0.8, 0.862, 0.869,0.8686, 0.8979,0.883,0.8887]

el_perfs = np.array([0.80,0.8301, 0.864,0.835,0.873,0.855,0.861, 0.838])
el_items = np.array([2, 4, 6,8,10,12, 14, 16])

plt.plot(el_items*5, el_perfs, 'r.-', label="Ellipses")
plt.plot(layers*7, layers_perfs, 'b.-', label="Gratings")
plt.xlabel("Number of free parameters")
plt.ylabel("Figure of merit")
plt.tight_layout()
plt.legend()
plt.savefig("Perf.png")
plt.savefig("Perf.pdf")
