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
    file = f"{folder}/free_pixmap_{i}/best.npz"
    if os.path.isfile(file):
        ds.append((f"PSO.{i}", np.load(file)))
    else:
        print(f"[WARNING]: broke at {i}")
        break
#layers = int(folder.split("_")[2])
def densify(X):
    X = np.array(X)
    print("X", X.shape)
    Y = np.zeros(int(np.max(X[:, 0])+1))
    for i, x in enumerate(X[:, 0]):
        x = int(x)
        Y[x] = X[i, 1]
    return Y
profiles = list()
if "cma" in folder:
    for i in range(len(ds)):
        profiles.append(densify(ds[i][1]["profile"]))
else:
    for i in range(len(ds)):
        profiles.append(-(ds[i][1]["profile"]))
ds = tqdm(ds)
dataset = np.asarray([extend(prof, 750) for (name, d), prof in zip(ds, profiles) ])
print("Loaded dataset of shape", dataset.shape)
configs = np.asarray([d["bd"] for name, d in ds ])
mean = np.mean(dataset, axis=0)
std = np.std(dataset, axis=0)
min = np.min(dataset, axis=0)
max = np.max(dataset, axis=0)
best = np.argmax(dataset[:, -1])
import numpy as np
print("BEST RESULT:", best, round(dataset[best, -1], 2), round(np.mean(dataset[:, -1]), 2))
if len(sys.argv) > 1 and sys.argv[3] == "fig":
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
ax.set_title(f"T[{best}]={100*dataset[best, -1]:1.3f}%")
plt.xlabel("Iterations")
plt.ylabel("Metric [Transmission of selected order]")
plt.axhline(0.8, color="k", ls=":")
plt.axhline(0.9, color="k", ls=":")
plt.plot(mean, color="k")
#plt.plot(min, color="k")
plt.plot(max, color="k")
#plt.plot(-(mean+std), color="k", ls=":")
#plt.plot(-(mean-std), color="k", ls=":")
fig.savefig(f"figs/convprof.png")
plt.close()

