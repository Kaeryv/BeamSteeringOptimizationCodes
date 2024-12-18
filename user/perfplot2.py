import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys
import os
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


def load_experiment_folder(folder):
    ds = list()
    for i in range(0, 100):
        file = f"data/{folder}/free_pixmap_{i}/best.npz"
        if os.path.isfile(file):
            ds.append((f"PSO.{i}", np.load(file)))
        else:
            break

    return np.asarray([np.min(d["profile"])  for name, d in ds ])

folders = {"Ellipses" : [
        "ellipsisd32_1_16_pso",
        "ellipsisd32_6_16_pso",
        "ellipsisd32_8_16_pso",
        "ellipsisd32_10_16_pso", #running
        "ellipsisd32_12_16_pso",
        "ellipsisd32_14_16_pso",
        "ellipsisd32_16_16_pso", #running
        "ellipsisd32_18_16_pso", #running
        "ellipsisd32_20_16_pso",
        ],
        # "ellipses": [
        #     "ellipsis_1_16_pso",
        #     "ellipsis_2_16_pso",
        #     "ellipsisold_6_16_pso",
        #     "ellipsisold_8_16_pso",
        #     "ellipsisold_10_16_pso",
        #     "ellipsisold_12_16_pso",
        #     "ellipsisold_14_16_pso",
        #     #"ellipsis_16_16_pso", redo
        #     "ellipsis_18_16_pso",
        #     #"ellipsis_20_16_pso",
        #     ],
        "Ellipses + Material": [
            "ellipsismm_2_16_pso",
            "ellipsismm_4_16_pso",
            "ellipsismm_6_16_pso",
            "ellipsismm_8_16_pso",
            "ellipsismm_10_16_pso",
            "ellipsismm_12_16_pso",
            "ellipsismm_14_16_pso",
            "ellipsismm_16_16_pso",
            "ellipsismm_18_16_pso",
            "ellipsismm_20_16_pso",
            ],
        "Gratings Stack": [
            "gratings_4_4_pso",
            "fftlike_8_8_pso",
            "fftlike_10_10_pso",
            "fftlike_12_12_pso",
            "fftlike_14_14_pso",
            "fftlike_16_16_pso",
            "fftlike_18_18_pso",
            "fftlike_20_20_pso",
            "gratings2_32_32_pso",
            ]
        }
num_params = [5, 5, 6, 7]
plot_data = list()
for (f, samples), p in zip(folders.items(), num_params):
    plot_data.append([])
    print(f"Experiment {f}")
    for sample in samples:
        ni = int(sample.split("_")[1])
        dataset = - load_experiment_folder(sample)
        mask = dataset < 0.5
        dataset = dataset[~mask]
        best = np.max(dataset)
        mean = np.mean(dataset)
        std = np.std(dataset) 
        print(sample, len(dataset), best)
        plot_data[-1].append((best, mean, std, p*ni))

fmts = ["r-o", "g-o", "b-o", 'k-o']
fig, ax = plt.subplots(figsize=(4.6,4))
for d, fmt, name in zip(plot_data, fmts, folders.keys()):
    d = np.array(d)
    #plt.scatter(d[:, -1], d[:, 0])
    #plt.errorbar(d[:, -1], d[:, 1], yerr=d[:, 2], fmt=fmt, ecolor = "black")
    ax.plot(d[:, -1], d[:, 0], fmt, label=name)
ax.set_xlabel("Number of free parameters")
ax.set_ylabel("Figure of merit")
plt.tight_layout()
plt.legend()
fig.savefig("figs/Perf.png")
plt.savefig("figs/Perf.pdf")
