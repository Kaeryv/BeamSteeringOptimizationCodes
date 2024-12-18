import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys
import os
def extend(x, nl):
    l = len(x)
    y = np.ones(nl) * x[-1]
    y[0:l] = x
    return y

folder = sys.argv[1]
count = int(sys.argv[2])
ds = list()
for i in trange(0, count):
    file = f"data/{folder}/free_pixmap_{i}/best.npz"
    if os.path.isfile(file):
        ds.append((f"PSO.{i}", np.load(file)))
    else:
        print(f"[WARNING]: broke at {i}")
        break
#layers = int(folder.split("_")[2])
ds = tqdm(ds)
dataset = np.asarray([extend(d["profile"], 900) for name, d in ds ])
print(dataset.shape)
configs = np.asarray([d["bd"] for name, d in ds ])
mean = np.mean(dataset, axis=0)
std = np.std(dataset, axis=0)
min = np.min(dataset, axis=0)
max = np.max(dataset, axis=0)
best = np.argmax(-dataset[:, -1])
print("BEST RESULT:", best, -dataset[best, -1])
if len(sys.argv) > 1 and sys.argv[1] == "fig":
    pass
else:
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
ax.set_title(f"T[{best}]={-100*dataset[best, -1]:1.3f}%")
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

