from types import SimpleNamespace
import logging
from copy import deepcopy as cpy
import numpy as np
from bast.crystal import Crystal
from bast.expansion import Expansion
from bast.layer import Layer
from PIL import Image 
from ellipse import draw_ellipse
from functools import partial
import os
from itertools import product
from multiprocessing import Pool
import time

NUM_THREADS = int(os.environ["NCPUS"]) if "NCPUS" in os.environ else 1

def str2polar(name):
    if name == "X":
        return (1, 0)
    elif name == "Y":
        return (0, 1)
    elif name == "XY":
        return (1, 1)
    elif name == "RCP":
        return (1, -1j)
    elif name == "LCP":
        return (1, 1j)
    else:
        print("Error: unknown polarization name.")
        exit()


def pics_from_design(X, sim_args):
    nl = sim_args.num_layers
    ni = sim_args.num_items

    xys, axes, angles, depths = np.split(X, [ni*nl*2, ni*nl*4, ni*nl*5], axis=0)
    xys = xys.reshape((nl, ni, 2))    
    axes = axes.reshape((nl, ni, 2))    
    angles = angles.reshape((nl, ni))
    pics = []
    def _draw(x, y, a, b, alpha):
        draw_ellipse(canvas, (x, y, a, b, alpha))
    for i, (xy, ax, ang, d) in enumerate(zip(xys, axes, angles, depths)):
        w, h = 256, 256
        canvas = np.zeros((h, w))
        for (x,y), (a, b), alpha in zip(xy, ax, ang):
            _draw(x,   y,   a, b, alpha)
            _draw(x-1, y,   a, b, alpha)
            _draw(x+1, y,   a, b, alpha)
            _draw(x,   y+1, a, b, alpha)
            _draw(x-1, y+1, a, b, alpha)
            _draw(x+1, y+1, a, b, alpha)
            _draw(x,   y-1, a, b, alpha)
            _draw(x-1, y-1, a, b, alpha)
            _draw(x+1, y-1, a, b, alpha)
            
        canvas = sim_args.elow + canvas * (sim_args.ehigh-sim_args.elow)
        pics.append(canvas)
    return pics

def solve(design,sim_args, twist_angle, polar, pics):
    polar = str2polar(polar)
    e1 = Expansion(sim_args.pw)
    e2 = Expansion(sim_args.pw)
    e2.rotate(twist_angle)
    etw = e1 + e2
    cl = Crystal.from_expansion(etw)

    nl = sim_args.num_layers
    ni = sim_args.num_items

    xys, axes, angles, depths = np.split(design, [ni*nl*2, ni*nl*4, ni*nl*5], axis=0)
    xys = xys.reshape((nl, ni, 2))    
    axes = axes.reshape((nl, ni, 2))    
    angles = angles.reshape((nl, ni))
    device = []
    cl.add_layer(f"Sref", Layer.half_infinite(e1, "reflexion", 1.0), True)
    cl.add_layer(f"Strans", Layer.half_infinite(e2, "transmission", 1.0), True)

    for i, (pic, d) in enumerate(zip(pics, depths)):
        curexp = (e1, e2)[i < nl // 2]
        cl.add_layer(f"L{i}", Layer.pixmap(curexp, pic, d), True)
        device.append(f"L{i}")
    
    cl.set_source(sim_args.wavelength, polar[0]**2, polar[1]**2, 0, 0)
    cl.set_device(device, [False]*len(device))
    cl.solve()
    R, T = cl.poynting_flux_end(only_total=False)
    return R, T, cl.expansion.g_vectors

def worker(config, design, sim_args, pics):
    polar, twist_angle = config
    _, T, kxy = solve(design, sim_args, twist_angle, polar, pics)
    return np.abs(T[1]).reshape(sim_args.pw[0]**2, sim_args.pw[1]**2), kxy

def main(sim_args, design, figpath):
    sim_args.angles = np.linspace(
        sim_args.angles[0], sim_args.angles[1], sim_args.angles[2]
    )
    pics = pics_from_design(design, sim_args)

    workerp = partial(
        worker, design=design, sim_args=sim_args, pics=cpy(pics)
    )
    # Processing
    configurations = list(product(sim_args.polarizations, sim_args.angles))
    angle_magnitudes, angle_gvectors = [], []
    start = time.time()
    with Pool(NUM_THREADS) as p:
        for magnitudes, gvectors in p.imap(workerp, configurations):
            angle_magnitudes.append(magnitudes)
            angle_gvectors.append(gvectors)
    stop = time.time()
    logging.info(f"Performed job in {(stop-start)}s.")
    angle_magnitudes = np.asarray(angle_magnitudes).reshape(
        len(sim_args.polarizations),
        len(sim_args.angles),
        *angle_magnitudes[0].shape,
    )
    angle_gvectors = np.asarray(angle_gvectors).reshape(
        len(sim_args.polarizations), len(sim_args.angles), *angle_gvectors[0].shape
    )

    c = (sim_args.pw[0]**2 - 1) // 2

    #metric = angle_magnitudes[..., c + sim_args.target_order[0], c + sim_args.target_order[1]]
    metric = angle_magnitudes.reshape((-1,81))[..., 8]
    # Its plot time
    if figpath is not None:
        import matplotlib.pyplot as plt
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
        ax1.plot(sim_args.angles, metric.flatten())
        im = ax2.matshow(np.vstack((np.hstack((pics[0],pics[1])))), cmap="Blues", extent=[0,2,0,3]) #np.hstack((pics[4],pics[5])) np.hstack((pics[2],pics[3]))
        ax2.axis("equal")
        ax3.scatter(angle_gvectors[0,8,0], angle_gvectors[0,8,1], color="black", facecolors='none', s=80)
        ax3.scatter(angle_gvectors[0,8,0, 8], angle_gvectors[0,8, 1, 8], color='red', s=80, facecolors='none')
        ax3.scatter(angle_gvectors[0,8,0], angle_gvectors[0,8,1], s=10*angle_magnitudes[0,8].flatten())
        plt.colorbar(im)
        fig.savefig(figpath)
        
    return metric

def __run__(design, sim_args, figpath=None):
    sim_args = (
        SimpleNamespace(**sim_args)
        if not isinstance(sim_args, SimpleNamespace)
        else sim_args
    )
    logging.info(f"{len(design)=}{figpath=}")
    r = main(cpy(sim_args), cpy(design), figpath)
    return {"fitness": np.mean(r)}


def __requires__():
    return {"variables": [ "design", "angles", "sim_args", "figpath"]}
