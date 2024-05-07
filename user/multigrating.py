import sys
sys.path.append(".")

# Disable multithreading
# We use EP workload instead
import os

from multiprocessing import Pool
NUM_THREADS = int(os.environ['NCPUS']) if 'NCPUS' in os.environ else 1


from bast.layer import Layer
from bast.extension import ExtendedLayer as EL
from bast.expansion import Expansion
from bast.crystal import Crystal
from user.charts import grating_summary_plot


from user.commons import freq2pix
from bast.draw import Drawing

import numpy as np
import logging
from types import SimpleNamespace
from functools import partial

import user.parameterization as prm



def binary_dilation(X, w=1):
    out = np.zeros_like(X)
    for i in range(len(X)):
        if X[i]:
            out[i-w:i+w+1] = True
    return out

def binary_erosion(X, w=1):
    out = np.zeros_like(X)
    for i in range(len(X)):
        if np.all(X[i-w:i+w+1]):
            out[i] = True 
    return out

def grating_filtering(X, width=1):

    bx = X > 2.5
    bx = binary_erosion(bx, width)
    bx = binary_dilation(bx, width)
    bx = np.logical_not(bx)
    bx = binary_erosion(bx, width)
    bx = binary_dilation(bx, width)
    bx = np.logical_not(bx)

    return 2 * (bx + 1)

fwidth=2
def build_substack(expansion, X, depths):
    # Create a crystal in the void with the array of gratings
    crystal = Crystal.from_expansion(expansion, void=True)

    device = list()
    for i, depth in enumerate(depths):
        d = Drawing((256,1), 2)
        bx = grating_filtering(X[i], width=fwidth)
        #print(f"{np.max(bx)}, {np.min(bx)}")
        #bx = X[i]
        d.from_numpy((bx).copy())
        if len(d.islands()) == 0 or np.all(X[i] == 2.0):
            crystal.add_layer_uniform(f"G{i}", 2.0,  depth)
        elif np.all(X[i] == 4.0):
            crystal.add_layer_uniform(f"G{i}", 4.0,  depth)
        else:
            crystal.add_layer_analytical(f"G{i}", d.islands(), d.background,  depth)
            #crystal.add_layer_pixmap(f"G{i}", d.canvas(),  depth)

        device.append(f"G{i}")

    crystal.set_device(device)
    crystal.fields = False
    return crystal


def solve_multigrating(X, depths, wl, twist_angle, polar=(1, 1), pw=(7,1)):
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(twist_angle)
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


def worker(twist_angle, gratings, depths, wl, pw, polar):
  _, T, kxy = solve_multigrating(gratings, depths, wl, twist_angle, pw=pw, polar=polar)
  return np.abs(T[1]).reshape(pw[0], pw[0]), kxy

def main(program, sim_args, design, angles=None, figpath=None):
    kp = (0, 0)

    if program == "angle_batch":
        pw = sim_args.pw
        if angles is None:
            angles = np.linspace(0, 60, 30)
        else:
            angles = np.linspace(angles[0], angles[1], angles[2])

        gratings, depthss = getattr(prm, sim_args.parameterization)(design, sim_args.elow, sim_args.ehigh, sim_args.bilayer_mode, **sim_args.parameterization_args)
        
        i_display = 5
        c = (pw[0] - 1) // 2
        workerp = partial(worker, gratings=gratings, depths=depthss, wl=sim_args.wavelength, pw=pw, polar=sim_args.polar)
        angle_magnitudes, angle_gvectors = [],[]
        with Pool(NUM_THREADS) as p:
            for magnitudes, gvectors in p.imap(workerp, angles):
                angle_magnitudes.append(magnitudes)
                angle_gvectors.append(gvectors)
        
        mag = np.asarray(angle_magnitudes)
        mag_display = np.copy(angle_magnitudes[i_display])
        gvectors_display = np.copy(angle_gvectors[i_display])
        metric = mag[:, c - 1, c + 1] #/ np.sum(mag, axis=(1,2))
        #print(f"{np.mean(metric)}")


        fom_percent = np.mean(metric*100)
        logging.debug(f"Grating sim got {fom_percent}% concentration.")


        if figpath is not None:
            bilayer_depth = np.sum(depthss, axis=1)
            gratings_picture = np.vstack((
                np.repeat( [grating_filtering(g, width=fwidth) for g in gratings[0]], 
                          np.round(100 * depthss[0] / np.sum(depthss[0])).astype(int), axis=0),
                np.repeat( [grating_filtering(g, width=fwidth) for g in gratings[1]], 
                          np.round(100 * depthss[1] / np.sum(depthss[1])).astype(int), axis=0),
            ))
            grating_summary_plot(figpath, angles, metric, gvectors_display, mag_display, (angles[i_display], 100*metric[i_display]), fom_percent, gratings_picture, bilayer_depth)
        return metric


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "help" or sys.argv[1] == "h":
        print("---------------------------------------------------------------")
        print("Multigrating simulator help")
        print("---------------------------------------------------------------")
        print("program 'angle_batch' takes 2 arguments: (input, figpath).")
        print("---------------------------------------------------------------")
        exit(0)
    

    assert len(sys.argv) > 2, "Missing input file."
    program = sys.argv[1]
    filepath = sys.argv[2]
    figpath = sys.argv[3] if len(sys.argv) > 3 else None
    design = np.load(filepath)["bd"]



    sim_args = SimpleNamespace(
            elow = 2.0,
            ehigh = 4.0,
            wavelength = 1.01,
            bilayer_mode="free",
            num_layers=10,
            pw=(7,1),
            polar=(1,1),
            parameterization="fftlike",
            parameterization_args={"harmonics": [0.5,1,1.5]}
        )
    if len(sys.argv) > 4:
        sim_args.pw = (int(sys.argv[4]), 1)
    design = design.reshape(sim_args.num_layers, -1).copy()
    main(program, sim_args, design, figpath=figpath)


# Keever stuff
def __run__(program, design, angles, sim_args, figpath=None):
    sim_args = SimpleNamespace(**sim_args) if not isinstance(sim_args, SimpleNamespace) else sim_args
    sim_args.polar = (1, 1j)
    logging.debug(f"{program=}{design.shape=}{angles=}{figpath=}")
    design = design.reshape(sim_args.num_layers, -1).copy()
    angles = angles.copy()
    raw_fom = main(
        program,
        sim_args,
        design,
        angles=angles,
        figpath=figpath,
    )
    fom = np.mean(raw_fom) #+ np.prod(raw_fom)
    return fom


def __requires__():
    return {"variables": ["program", "design", "angles", "sim_args"]}
