import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys

def extend(x, nl):
    l = len(x)
    y = np.ones(nl) * x[-1]
    y[0:l] = x
    return y

folder = sys.argv[1]
ds = [ (f"PSO.{i}", np.load(f"data/{folder}/free_pixmap_{i}/best.npz")) for i in trange(0, 100) ]
layers = int(folder.split("_")[2])
ds = tqdm(ds)
dataset = np.asarray([extend(d["profile"], 500) for name, d in ds ])
mean = np.mean(dataset, axis=0)
std = np.std(dataset, axis=0)
min = np.min(dataset, axis=0)
max = np.max(dataset, axis=0)
best = np.argmax(-dataset[:, -1])
print("BEST RESULT:", best, -dataset[best, -1])
exit()
fig, ax = plt.subplots(figsize=(5,4))#, dpi=250
if False:
    for name, d in ds:
        profile = d["profile"]
        if len(profile.shape) == 2:
            iterations = profile[:, 0]
            profile = profile[:, 1]
        else:
            iterations = np.arange(len(profile))
        if np.any(profile<0):
            profile *= -1
        ax.plot(iterations, profile, color="orange", marker=".", ls=None,label=f"{name}", alpha=0.01)
ax.set_title(f"T[{best}]={-50*dataset[best, -1]:1.3f}%")
plt.xlabel("Iterations")
plt.ylabel("Metric [Transmission of selected order]")
plt.axhline(0.8, color="k", ls=":")
plt.axhline(0.9, color="k", ls=":")
plt.plot(-mean, color="k")
plt.plot(-min, color="k")
plt.plot(-max, color="k")
plt.plot(-(mean+std), color="k", ls=":")
plt.plot(-(mean-std), color="k", ls=":")
fig.savefig(f"figs/convprof.png")
plt.close()

