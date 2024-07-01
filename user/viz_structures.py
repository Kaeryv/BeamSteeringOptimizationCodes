import sys 
sys.path.append(".")

import os
from glob import glob
import numpy as np

from user import parameterization as prm
from user.charts import grating_side_picture
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, FeatureAgglomeration, AgglomerativeClustering
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.colors as cm
import matplotlib
from matplotlib.gridspec import GridSpec
import pickle



def find_longest_island(x):
    lengths = list()
    for xo in x:
        idx_pairs = np.where(np.diff(np.hstack(([False],xo==4.0,[False]))))[0].reshape(-1,2)
        if np.all(xo == 4.0) or np.all(xo == 2.0):
            lengths.append(0)
            continue
        lengths.append(np.max(np.diff(idx_pairs,axis=1)))
    return np.argmax(lengths)

def align_gratings(canvases):
    ref = canvases[0].copy()
    xsize = ref.shape[1]
    for i, cnv in enumerate(tqdm(canvases)):
        metric = []
        for j in range(xsize):
            cnvr = np.roll(cnv, j, axis=-1)
            diff = np.mean(np.power(cnvr-ref, 2))
            metric.append(diff)

        rolls = np.argmin(metric)

        fig, (ax1,ax2) = plt.subplots(2)
        ax1.matshow(canvases[i], cmap="Blues")
        canvases[i] = np.roll(canvases[i], rolls, axis=-1)
        ref = (i+1) / (i+2) * ref + canvases[i] / (i+2)
        ax2.matshow(canvases[i], cmap="Blues")
        fig.savefig(f"ba_{i}.png")
        plt.close()

    return canvases

lnum_items = [6,8,10,12,14,16]
#gratings_layers =[8,10]
#roots = [(f"data/el{num_items}_v3_16_pso/free_pixmap_*/best.npz", "ellipsis") for num_items in lnum_items]
##roots.extend([(f"data/gratings_v3_{num_layers}_pso/free_pixmap_*/best.npz", "fftlike") for num_layers in gratings_layers])
##lnum_items.extend(gratings_layers)
#files = list()
lsnum_items = list()
#types = list()
#for (root, type), ni in zip(roots, lnum_items):
#    fl = glob(root)
#    files.extend(fl)
#    types.extend([type]*len(fl))
#    lsnum_items.extend([ni]*len(fl))
#
#designs = list()
#fitness = list()
#for f in files:
#    buf = np.load(f)
#    designs.append(buf["bd"])
#    fitness.append(buf["bf"])

if __name__ == "__main__":
    with open("raw_designs.pkl", 'rb') as f:
        raw = pickle.load(f)
        designs = raw["designs"]
        fitness = raw["fitness"]
    fitness = np.asarray(fitness)
    for l in lnum_items:
        lsnum_items.extend([l]*101)
    if True:
        # Generate pictures from devices
        gratings_pictures = list()
        depths = list()
        for d, ni in zip(designs, lsnum_items):
            type = "ellipsis"
            if type == "ellipsis":
                gratings, layers_depths = getattr(prm, "ellipsis")(
                    d,
                    2.0,
                    4.0,
                    "free",
                    num_items=ni,
                    num_layers=16
                )
            else:
                gratings, layers_depths = getattr(prm, "fftlike")(
                    d,
                    2.0,
                    4.0,
                    "free",
                    num_layers=ni,
                    harmonics = [0.5, 1, 1.5]
                )
            depth = np.sum(layers_depths)
            depths.append(depth)
            gratings_picture, bilayer_depth = grating_side_picture(gratings, layers_depths, 2,yres=256)
            gratings_pictures.append(gratings_picture)#gratings_picture
        gratings_pictures = np.asarray(gratings_pictures)
        depths = np.asarray(depths)
        N = gratings_pictures.shape[0]
        img_shape = gratings_pictures.shape[1:]
'''
designs = list()
fitness = list()
for f in files:
    buf = np.load(f)
    designs.append(buf["bd"])
    fitness.append(buf["bf"])
import pickle

with open("raw_designs.pkl", 'wb') as f:
    pickle.dump({"designs": designs, "fitness":fitness}, f)
exit()
fitness = np.asarray(fitness)
import matplotlib.colors as cm
import matplotlib
if True:
    # Generate pictures from devices
    gratings_pictures = list()
    depths = list()
    for d, ni, type in zip(designs, lsnum_items, types):
        if type == "ellipsis":
            gratings, layers_depths = getattr(prm, "ellipsis")(
                d,
                2.0,
                4.0,
                "free",
                num_items=ni,
                num_layers=16
            )
        else:
            gratings, layers_depths = getattr(prm, "fftlike")(
                d,
                2.0,
                4.0,
                "free",
                num_layers=ni,
                harmonics = [0.5, 1, 1.5]
            )
        depth = np.sum(layers_depths)
        depths.append(depth)
        gratings_picture, bilayer_depth = grating_side_picture(gratings, layers_depths, 2,yres=256)
        gratings_pictures.append(gratings_picture)#gratings_picture
    gratings_pictures = np.asarray(gratings_pictures)
    depths = np.asarray(depths)
    N = gratings_pictures.shape[0]
    img_shape = gratings_pictures.shape[1:]
    gratings_pictures = align_gratings(gratings_pictures)
    #exit()
    xs = gratings_pictures.reshape(N, -1)
    xmean = xs.mean()
    xstd = xs.std()
    xs -= xmean
    xs[:,xstd>0.0] /= xstd[xstd>0.0]
    np.savez_compressed("mldb.npz", xs=xs, y=fitness,img_shape=img_shape)
    #exit()
    model = PCA(2)
    #model = TSNE(2, perplexity=400)
    #predictor = KMeans(9)
    predictor = AgglomerativeClustering(9)
    labels = predictor.fit_predict(xs)
    x = model.fit_transform(xs)
    fig, (ax1,ax2) = plt.subplots(2)
'''
    
        if os.path.isfile("gratings_cache.npy"):
            gratings_pictures = np.load("gratings_cache.npy")
        else:
            gratings_pictures = align_gratings(gratings_pictures)
            np.save("gratings_cache.npy",gratings_pictures)
        #exit()
        xs = gratings_pictures.reshape(N, -1)
        xmean = xs.mean()
        xstd = xs.std()
        xs -= xmean
        xs[:,xstd>0.0] /= xstd[xstd>0.0]
        np.savez_compressed("mldb.npz", xs=xs, y=fitness,img_shape=img_shape)
        #exit()
        NPCA=6
        model = PCA(NPCA)
        #model = TSNE(2, perplexity=400)
        #predictor = KMeans(9)
        N = 7
        predictor = AgglomerativeClustering(N)
        x = model.fit_transform(xs)
        labels = predictor.fit_predict(x)
    
        fig = plt.figure(figsize=(10,10))
        gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax4 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])
        colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(N)]
        average_fitness = [
                np.mean(fitness[labels==i]) for i in range(N)
                ]
    
        def plot_circle(ax, x, y, r):
            t = np.linspace(0, 2*np.pi, 128)
            ax.plot(x+r*np.cos(t),y+r*np.sin(t),'k-')
            
        ordering = np.argsort(average_fitness)
        colors = list(reversed(colors))
        for i, col in zip(range(N), colors):
            ax1.scatter(*x[labels==ordering[i]].T[0:2], c=col, s=20)
        ax4.scatter(*x.T[2:4], c=fitness, cmap="RdBu", alpha=0.5, s=20)
        best_centroid = np.mean(x[labels==ordering[i]], axis=0)
        plot_circle(ax4, best_centroid[0], best_centroid[1], 10)
        ax1.axis("square")
        ax4.axis("square")
    
        bins = np.linspace(0.65,0.9, 20)
        for i, col in zip(range(N), colors):
            ax3.hist(fitness[labels==ordering[i]], bins=bins,bottom=10*(i+1), color=col,alpha=0.5, density=True)
        ax3.set_ylim(0, 90)
        best_centroid += np.random.rand(NPCA)-0.5
        centroid_recons = model.inverse_transform(best_centroid).reshape(img_shape)
        centroid_recons *= xstd
        centroid_recons += xmean
        mid = 3
        centroid_recons[centroid_recons>mid]=4
        centroid_recons[centroid_recons<=mid]=2
        centroid_recons= np.roll(centroid_recons, 12, axis=1)
        ss = 32
        im = ax2.matshow(centroid_recons[::ss], cmap="Blues", extent=[0,1,0,4])
        ax2.axis("equal")
        ax2.set_title(str(centroid_recons[::ss].shape))
        np.save("candidate.npy", centroid_recons[::ss])
        plt.colorbar(im)
    
        fig.savefig("PCA.png")
        matplotlib.rcParams['axes.linewidth'] = 3
    
        fig, axs = plt.subplots(1,N, figsize=(9,3))
        axs = axs.flatten()
    
        for i, (ax,col) in enumerate(zip(axs, colors)):
            if i >=9:
                break
            img = gratings_pictures[labels==i].mean(0)
            _ = [e.set_edgecolor(col) for e in ax.spines.values()]
            ax.matshow(img, extent=[0, 1, 0, depths[0]])
            #ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.savefig("Clusters.png")
        fig.savefig("Clusters.pdf")
    
    
    
        #fig, axs = plt.subplots(5, 5, figsize=(9,9))
        #theta = np.linspace(0, 2*np.pi, 25)
        #for t, ax in zip(theta, axs.flat):
        #    latent = [200*np.cos(t), 200*np.sin(t)]
        #    image = model.inverse_transform(latent)
        #    ax.matshow(image.reshape(img_shape), extent=[0,1, 0, depths[0]])
        #axs = axs.flatten()
        #fig.savefig("Recons.png")
        # fig.savefig("Recons.pdf")
    #harmonics = np.asarray([0.5, 1, 1.5])
    
    if False: # Histogram of device height
        designs = np.reshape(designs, (len(designs), layers, -1))
        depths = designs[:, :, -1].sum(-1)
        fig, (ax1,ax2) = plt.subplots(2)
        ax1.hist(depths)
        ax2.plot(fitness, depths, 'r.')
        fig.savefig("figs/depths.png")
        print("mean depth", depths.mean())
        exit()
    
    if False:
        designs = np.reshape(designs, (len(designs), 16, -1))
        sort = np.argsort(-fitness)
        fig, ax = plt.subplots()
        for j in range(600):
            i = sort[j]
            amps, phases, depths = np.split(designs[i, :], [3,6], axis=-1)
            zs = amps * np.exp(2j*np.pi*(phases))
            zs = zs[:, 1]
            ax.scatter(zs.real, zs.imag, marker="o", s=40*depths[:,0], alpha=0.1, c=[fitness[i]/np.max(fitness)]*16, vmin=0, vmax=1)
        plt.savefig("z.png")
        exit()
    
    if False: # 2D viz and clustering
        cluster = KMeans(3)
        scaler = StandardScaler()
        #model = PCA(2)
        model = FeatureAgglomeration(2)
        #model = TSNE(2, perplexity=400)
        xs = scaler.fit_transform(designs)
        labels = cluster.fit_predict(xs)
        x = model.fit_transform(xs)
        fig, (ax1,ax2) = plt.subplots(2)
        ax1.scatter(*x.T, c=fitness)
        ax2.scatter(*x.T, c=labels, cmap="tab10")
        for l in np.unique(labels):
            print(np.mean(fitness[labels==l]))
        fig.savefig("viz.png")
