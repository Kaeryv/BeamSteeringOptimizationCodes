import sys
sys.path.append(".")

import os
from multiprocessing import Pool

NUM_THREADS = int(os.environ["NCPUS"]) if "NCPUS" in os.environ else 1
print("Using ", NUM_THREADS, "threads.")

from bast.layer import Layer
from bast.extension import ExtendedLayer as EL
from bast.expansion import Expansion
from bast.crystal import Crystal
from user.charts import (
        grating_summary_plot,
        grating_side_picture,
        plot_angle_magnitude)

from user.commons import freq2pix
from bast.draw import Drawing

import numpy as np
import logging
from types import SimpleNamespace
from functools import partial

import user.parameterization as prm

from itertools import product

fwidth = 2
import pickle

def dump_variable(variable, filename):
    with open(filename, 'wb') as file:
        pickle.dump(variable, file)
import matplotlib.pyplot as plt
def build_substack(expansion, X, depths):
    # Create a crystal in the void with the array of gratings
    crystal = Crystal.from_expansion(expansion, void=True)
    device = list()
    for i, depth in enumerate(depths):
        d = Drawing((512, 1), 2)
        elow = np.min(X[i])
        ehigh = np.max(X[i])
        if not np.all(X[i] == X[i, 0]):
            pass
            bx = X[i]
            #bx = prm.grating_filtering(X[i], width=fwidth, elow=elow, ehigh=ehigh)
        else:
            bx = X[i]
        if np.all(X[i] == X[i, 0]):
            crystal.add_layer_uniform(f"G{i}", X[i, 0], depth)
        else:
            #d.from_numpy((bx).copy())
            #len(d.islands()) == 0 or n
            #crystal.add_layer_analytical(f"G{i}", d.islands(), d.background, depth)
            crystal.add_layer_pixmap(f"G{i}", bx[:, np.newaxis],  depth)

        device.append(f"G{i}")

    crystal.set_device(device)
    crystal.fields = False
    return crystal


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


def solve_multigrating(X, depths, wl, twist_angle, polar=(1, 1), pw=(7, 1)):
    if isinstance(polar, str):
        polar = str2polar(polar)
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(twist_angle/2)
    e1.rotate(-twist_angle/2)
    etw = e1 + e2

    # Build the outer medium (air)
    lr = EL(etw, Layer.half_infinite(e1, "reflexion", 1.0))
    lr.fields = True
    lt = EL(etw, Layer.half_infinite(e2, "transmission", 1.0))
    lt.fields = True

    # Build the upper and lower stacks
    X1, X2 = X
    depths1, depths2 = depths
    crystal_up = build_substack(e1, X1, depths1)
    crystal_do = build_substack(e2, X2, depths2)

    # Now everything is defined. Let's create the crystal
    big_crystal = Crystal.from_expansion(etw)
    big_crystal.add_layer("Sref", lr)
    big_crystal.add_layer("S1", EL(etw, crystal_up))
    big_crystal.add_layer("S2", EL(etw, crystal_do))
    big_crystal.add_layer("Strans", lt)
    big_crystal.set_device(["S1", "S2"])

    big_crystal.set_source(wl, polar[0], polar[1], 0, 0)
    big_crystal.solve()
    R, T = big_crystal.poynting_flux_end(only_total=False)
    return R, T, big_crystal.expansion.g_vectors


def worker(config, gratings, depths, wl, pw):
    polar, twist_angle = config
    _, T, kxy = solve_multigrating(
        gratings, depths, wl, twist_angle, pw=pw, polar=polar
    )
    return np.abs(T[1]).reshape(pw[0], pw[0]), kxy



def main(sim_args, design):
    pw = sim_args.pw
    sim_args.angles = np.deg2rad(
        np.linspace(
            sim_args.angles[0], sim_args.angles[1], sim_args.angles[2]
        )
    )
    configurations = list(product(sim_args.polarizations, sim_args.angles))

    # From reduced parameters to simulation parameters
    gratings, depthss = getattr(prm, sim_args.parameterization)(
        design,
        sim_args.elow,
        sim_args.ehigh,
        sim_args.bilayer_mode,
        **sim_args.parameterization_args,
    )

    workerp = partial(
        worker, gratings=gratings, depths=depthss, wl=sim_args.wavelength, pw=pw
    )

    angle_magnitudes, angle_gvectors = [], []
    with Pool(NUM_THREADS) as p:
        for magnitudes, gvectors in p.imap(workerp, configurations):
            angle_magnitudes.append(magnitudes)
            angle_gvectors.append(gvectors)

    angle_magnitudes = np.asarray(angle_magnitudes).reshape(
        len(sim_args.polarizations),
        len(sim_args.angles),
        *angle_magnitudes[0].shape,
    )
    angle_gvectors = np.asarray(angle_gvectors).reshape(
        len(sim_args.polarizations), len(sim_args.angles), *angle_gvectors[0].shape
    )

    c = (pw[0] - 1) // 2
    metric = angle_magnitudes[..., c + sim_args.target_order[0], c + sim_args.target_order[1]]

    fom_percent = np.mean(metric * 100)
    logging.debug(f"Grating sim got {fom_percent}% concentration.")

    results = SimpleNamespace(
        orders_vectors=angle_gvectors.copy(),
        orders_transmission=angle_magnitudes.copy(),
        gratings=gratings,
        layers_depths=depthss,
        sim_args=sim_args,
        metric=metric,
    )
    return results

def summarygraph(figpath, r, displayed_angle):
    gratings_picture, bilayer_depth = grating_side_picture(
        r.gratings, r.layers_depths, fwidth
    )

    fig, axs = grating_summary_plot(
        figpath,
        r.sim_args.angles,
        r.metric[0],
        r.orders_vectors[0, displayed_angle],
        r.orders_transmission[0, displayed_angle],
        gratings_picture,
        bilayer_depth,
    )
    colors = ['red', 'black', 'blue', 'brown', 'green']
    for i, col, pol in zip(range(len(r.metric)), colors, r.sim_args.polarizations):
        plot_angle_magnitude(axs[0,0], r.sim_args.angles, r.metric[i], style={"color": col, "ls": "-.", "label": pol})
    fig.legend()
    fig.savefig(figpath, transparent=False)
    fig.savefig(figpath.replace(".png", ".pdf"), transparent=True)


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "help" or sys.argv[1] == "h":
        print("---------------------------------------------------------------")
        print("Multigrating simulator help")
        print("---------------------------------------------------------------")
        print("program 'angle_batch' takes 2 arguments: (input, figpath).")
        print("---------------------------------------------------------------")
        exit(0)

    assert len(sys.argv) > 2, "Missing input file."
    filepath = sys.argv[1]
    figpath = sys.argv[2] if len(sys.argv) > 2 else None
    if filepath.endswith(".npz"):
        design = np.load(filepath)["bd"]
    elif filepath.endswith(".npy"):
        design = np.load(filepath)
    print(design.shape)
    sim_args = SimpleNamespace(
        elow=2.0,
        ehigh=4.0,
        wavelength=1.01,
        bilayer_mode="free",
        num_layers=16,
        pw=(7, 1),
        polarizations=["X", "Y","RCP", "LCP"],#, "Y", "XY", "LCP", "RCP"],  # LCP+ RCP-
        angles=[0, 60, 91],
        parameterization="ellipsis",
        target_order=(-1, +1),
        #parameterization_args={"num_layers": 14, "harmonics": [0.5,1,1.5]},
        #parameterization_args={"num_items": 12, "num_layers": 16,"depth": 3.2},#,"materials": [2.0, 3.0, 4.0 ]
        parameterization_args={"num_items": 12, "num_layers": 16,"depth": 3.2, "materials": [2.0, 3.0, 4.0 ]},#,
    )   
    if len(sys.argv) > 4:
        sim_args.pw = (int(sys.argv[4]), 1)
    r = main(sim_args, design)
    dump_variable(r, "sim_data.pkl")
    summarygraph(figpath, r, 5)


from copy import deepcopy as cpy
# Keever stuff
def __run__(program, design, sim_args, figpath=None):
    sim_args = (
        SimpleNamespace(**sim_args)
        if not isinstance(sim_args, SimpleNamespace)
        else sim_args
    )
    logging.debug(f"{program=}{design=}{figpath=}")
    r = main(cpy(sim_args), cpy(design))
    if figpath is not None:
        summarygraph(figpath, r, 5)
    return {"fitness": np.mean(r.metric)}


def __requires__():
    return {"variables": ["program", "design", "angles", "sim_args", "figpath"]}
