"""
Keever executable module
"""


from types import SimpleNamespace
import os
from os.path import join
import numpy as np
import logging
from sklearn.decomposition import PCA
import pickle
from PIL import Image

def __run__(pca_opts, design, figpath=None, workdir="./tmp/"):
    pca_opts = SimpleNamespace(**pca_opts)
    if not os.path.isfile(pca_opts.modelpath):
        if os.path.isfile(pca_opts.dbpath):
            gratings_pictures = np.load(pca_opts.dbpath)
            logging.debug(f"Loaded database of {gratings_pictures.shape} items for PCA.")
        else:
            logging.error(f"File [{pca_opts.dbpath}] not found.")
            exit()
        
        # Prepare the dataset
        N = gratings_pictures.shape[0]
        im_shape = gratings_pictures.shape[1:]
        xs = gratings_pictures.reshape(N, -1)
        xmean = xs.mean()
        xstd = xs.std()
        xs -= xmean
        xs[:,xstd > 0.0] /= xstd[xstd > 0.0]
        model = PCA(pca_opts.NPC)
        xpc = model.fit_transform(xs)
        vmean = xpc.mean(axis=0)
        vstd = xpc.std(axis=0)
        with open(pca_opts.modelpath, "wb") as f:
            pickle.dump({"model": model, "bounds": (vmean, vstd, xmean, xstd, im_shape)}, f)
    else:
        with open(pca_opts.modelpath, "rb") as f:
            d = pickle.load(f)
            model = d["model"]
            vmean, vstd, xmean, xstd, im_shape = d["bounds"]

    logging.debug(f"{design.shape}")
    design *= vstd
    design += vmean
    X = model.inverse_transform(design).reshape(im_shape)
    X *= xstd
    X += xmean
    is_air = np.mean(X, axis=1) < 2.0
    X = X[~is_air,:]
    new_depth = 6 * X.shape[0] / im_shape[0]

    X = Image.fromarray(X)
    X = X.resize((256, pca_opts.num_layers), Image.Resampling.NEAREST)
    X = np.array(X)
    mid = 3
    X[X >  mid] = 4
    X[X <= mid] = 2

    return {"design": (X, new_depth)}


def __requires__():
    return {"variables": ["pca_opts", "workdir", "design", "figpath"]}
