import sys 
sys.path.append(".")
import matplotlib.colors as cm
from matplotlib.gridspec import GridSpec
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


import pickle
from PIL import Image

def find_longest_island(x):
    lengths = list()
    for xo in x:
        idx_pairs = np.where(np.diff(np.hstack(([False],xo==4.0,[False]))))[0].reshape(-1,2)
        if np.all(xo == 4.0) or np.all(xo == 2.0):
            lengths.append(0)
            continue
        lengths.append(np.max(np.diff(idx_pairs,axis=1)))
    return np.argmax(lengths)

def align_gratings(canvases, plot=False):
    ref = canvases[0].copy()
    xsize = ref.shape[1]
    for i, cnv in enumerate(tqdm(canvases)):
        metric = []
        for j in range(xsize):
            cnvr = np.roll(cnv, j, axis=-1)
            diff = np.mean(np.power(cnvr-ref, 2))
            metric.append(diff)

        rolls = np.argmin(metric)
        if plot:
            fig, (ax1,ax2) = plt.subplots(2)
            ax1.matshow(canvases[i], cmap="Blues")
            canvases[i] = np.roll(canvases[i], rolls, axis=-1)
            ref = (i+1) / (i+2) * ref + canvases[i] / (i+2)
            ax2.matshow(canvases[i], cmap="Blues")
            fig.savefig(f"ba_{i}.png")
            plt.close()
        else:
            canvases[i] = np.roll(canvases[i], rolls, axis=-1)
            ref = (i+1) / (i+2) * ref + canvases[i] / (i+2)


    return canvases

def loaddb_from_outputs(roots):
    list_files   = list()
    list_nitems  = list()
    list_nlayers = list()
    list_designs = list()
    list_fitness = list()
    list_types   = list()
    for root in roots:
         type = root.split("/")[1].split("_")[0]
         nitems = int(root.split("/")[1].split("_")[1])
         nlayers = int(root.split("/")[1].split("_")[2])
         optim = root.split("/")[1].split("_")[3]
         files = glob(root)
         list_files.extend(files)
         list_nitems.extend([nitems]*len(files))
         list_nlayers.extend([nlayers]*len(files))
         list_types.extend([type]*len(files))
    
    for f in list_files:
        buf = np.load(f)
        list_designs.append(buf["bd"])
        list_fitness.append(buf["bf"])

    return list_designs, list_fitness, list_nitems, list_nlayers, list_types
import umap

if __name__ == "__main__":
    if os.path.isfile("raw_designs.pkl"):
        with open("raw_designs.pkl", 'rb') as f:
            raw = pickle.load(f)
            list_designs, list_fitness, list_nitems, list_nlayers, list_types, gratings_pictures, fitness = [ 
                raw[e] for e in 'list_designs list_fitness list_nitems list_nlayers list_types gratings_pictures fitness'.split()
            ]
    else:
        roots =      [ f"data/ellipsis_{nitems}_16_pso/free_pixmap_*/best.npz" for nitems in [6,8,10,12,14,16,18]]
        roots.extend([ f"data/fftlike_3_{nlayers}_pso/free_pixmap_*/best.npz" for nlayers in [8,10,12,14,16,18,20]])#,20
        list_designs, list_fitness, list_nitems, list_nlayers, list_types = loaddb_from_outputs(roots)
        fitness = np.asarray(list_fitness)

        # Generate pictures from devices
        gratings_pictures = list()
        depths = list()
        bonus_args = {"ellipsis": {}, "fftlike": {"harmonics": [0.5, 1.0, 1.5]}, "ellipsisold": {} }
        for d, ni, type, nl in zip(tqdm(list_designs), list_nitems, list_types, list_nlayers):
            gratings, layers_depths = getattr(prm, type)(d, 2.0, 4.0, "free", num_items=ni, num_layers=nl, **bonus_args[type])
            depth = np.sum(layers_depths)
            depths.append(depth)
            gratings_picture, bilayer_depth = grating_side_picture(gratings, layers_depths, 2,ppm=32, pic_height=6)
            gratings_pictures.append(gratings_picture)
        gratings_pictures = np.asarray(gratings_pictures)
        depths = np.asarray(depths)

        gratings_pictures = align_gratings(gratings_pictures, plot=False)
        with open("raw_designs.pkl", 'wb') as f:
            pickle.dump({ key: globals()[key] for key in 'list_designs list_fitness list_nitems list_nlayers list_types gratings_pictures fitness'.split()}, f)
    
    np.save("gratings_cache.npy", gratings_pictures)
    # Prepare and normalize
    N = gratings_pictures.shape[0]
    img_shape = gratings_pictures.shape[1:]
    xs = gratings_pictures.reshape(N, -1)
    xmean = xs.mean()
    xstd = xs.std()
    xs -= xmean
    xs[:,xstd>0.0] /= xstd[xstd>0.0]

    # Create and fit models
    NPCA, NCLUSTERS = 2, 7
    #model = PCA(NPCA)
    model = umap.UMAP()

    predictor = AgglomerativeClustering(NCLUSTERS)
    x = model.fit_transform(xs)
    labels = predictor.fit_predict(x)

    # The figure
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax4 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(NCLUSTERS)]
    average_fitness = [ np.mean(fitness[labels==i]) for i in range(NCLUSTERS) ]

    def plot_circle(ax, x, y, r):
        t = np.linspace(0, 2*np.pi, 128)
        ax.plot(x+r*np.cos(t),y+r*np.sin(t),'k-')
    def reject_outliers(data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        data_new = data.copy()
        data_new[s>m] = np.nan
        return data_new
    
    ordering = np.argsort(average_fitness)
    colors = list(reversed(colors))
    pcslice = slice(0,2)
    for i, col in zip(range(NCLUSTERS), colors):
        ax1.scatter(*x[labels==ordering[i]].T[pcslice], c=col, s=20)
    ax4.scatter(*x.T[pcslice], c=reject_outliers(fitness), cmap="RdBu", alpha=0.5, s=20)
    #ax4.scatter(*x.T[pcslice], c=list_nitems, cmap="tab10", alpha=1.0, s=20)
    best_centroid = np.mean(x[labels==ordering[i]], axis=0)
    #plot_circle(ax4, best_centroid[0], best_centroid[1], 10)
    
    ax1.axis("square")
    ax4.axis("square")
    ax1.set_xticks([])
    ax4.set_xticks([])
    ax1.set_yticks([])
    ax4.set_yticks([])
    ax1.text(1,0, "UMAP: n_neighbors=15, min_dist=0.1", transform=ax1.transAxes, fontsize=12, ha='right', va='bottom')
    bins = np.linspace(0.65,0.9, 20)
    for i, col in zip(range(NCLUSTERS), colors):
        ax3.hist(fitness[labels==ordering[i]], bins=bins,bottom=10*(i+1), color=col,alpha=0.5, density=True)
    ax3.set_ylim(0, 90)
    if False:
        best_centroid += np.random.rand(NPCA)-0.5
        centroid_recons = model.inverse_transform(best_centroid).reshape(img_shape)
        centroid_recons *= xstd
        centroid_recons += xmean
        is_air = np.mean(centroid_recons, axis=1) < 2.0
        im_shape = centroid_recons.shape
        centroid_recons = centroid_recons[~is_air,:]
        new_depth = 6 * centroid_recons.shape[0] / im_shape[0]
        print(new_depth,centroid_recons.shape[0], im_shape[0])
        g = Image.fromarray(centroid_recons)
        centroid_recons = g.resize((256,16), Image.Resampling.NEAREST)
        centroid_recons = np.array(centroid_recons)
        mid = 3
        centroid_recons[centroid_recons>mid]=4
        centroid_recons[centroid_recons<=mid]=2
        im = ax2.matshow(centroid_recons, cmap="Blues", extent=[0,1,0,6])
        ax2.axis("equal")
        ax2.set_title(str(centroid_recons.shape))
        plt.colorbar(im)

    fig.savefig("figs/pca_full.png")
    fig.savefig("figs/pca_full.svg")

    fig, axs = plt.subplots(1, NCLUSTERS, figsize=(6,4))
    for i, (ax,col) in enumerate(zip(axs, colors)):
        if i >=9:
            break
        img = gratings_pictures[labels==i].mean(0)
        _ = [e.set_edgecolor(col) for e in ax.spines.values()]
        img *= xstd
        img += xmean
        im = ax.matshow(img, extent=[0, 1, 0, 6], cmap="Blues", vmin=1, vmax=4)
        ax.axis("equal")
        #plt.colorbar(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0,1)
        ax.set_ylim(0,6)
        [i.set_linewidth(3) for i in ax.spines.values()]
    fig.savefig("figs/pca_clusters.png")
    fig.savefig("figs/pca_clusters.svg")
'''
    
    
    
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
'''
