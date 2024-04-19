from bast.layer import Layer, stack_layers
from bast.extension import ExtendedLayer as EL
from bast.expansion import Expansion
from bast.crystal import Crystal
from .charts import grating_summary_plot

import sys

sys.path.append(".")

from user.commons import freq2pix

import numpy as np
import logging
import os

pw = (5, 1)
kp = (0, 0)


def build_substack(expansion, X, depths):
    # Create a crystal in the void with the array of gratings
    crystal = Crystal.from_expansion(expansion, void=True)

    device = list()
    for i, depth in enumerate(depths):
        crystal.add_layer_pixmap(f"G{i}", X[i, :, np.newaxis], depth)
        device.append(f"G{i}")

    crystal.set_device(device)
    crystal.fields = False
    return crystal


def solve_multigrating(X, depths, wl, twist_angle, polar=(1, 1)):
    R, T = [], []
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(twist_angle)
    etw = e1 + e2

    lr = EL(etw, Layer.half_infinite(e1, "reflexion", 1.0))
    lr.fields = True

    X1, X2 = X
    depths1, depths2 = depths
    # Build the upper crystal
    crystal_up = build_substack(e1, X1, depths1)
    crystal_do = build_substack(e2, X2, depths2)

    lt = EL(etw, Layer.half_infinite(e2, "transmission", 1.0))
    lt.fields = True

    # Now everything is defined. Let's create the big crystal
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


def main(program, design=None, angles=None, figpath=None, bilayer_mode="copy", wavelength=1.01):
    epsg = 4
    epslow = 2

    if program == "showme":
        N = 8
        amps = np.random.rand(N, 5)
        phases = np.random.rand(N, 5) * 2 * np.pi
        g = np.asarray([freq2pix(a, p)[1] for a, p in zip(amps, phases)])
        g = (1 + g) * epsg
        depths = np.random.rand(N)
        R, T, kxy = solve_multigrating((g, g), depths, wavelength, 3)
        mag = np.abs(T[1])
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #gratings_depth = np.repeat(
        #    g, np.round(100 * depths / np.sum(depths)).astype(int), axis=0
        #)
        #ax1.matshow(gratings_depth, cmap="Blues")
        #ax2.scatter(*kxy, s=100, marker="o", facecolors="none", edgecolors="gray")
        #ax2.scatter(*kxy, s=mag / np.max(mag) * 100, c="r")
        #ax1.axis("equal")
        #fig.savefig("figs/multigrating.png")

    elif program == "angle_batch":
        if angles is None:
            angles = np.linspace(0, 60, 30)
        else:
            angles = np.linspace(angles[0], angles[1], angles[2])

        amps, phases, depths = np.split(design, [5, 10], axis=1)
        depths = np.squeeze(depths)
        phases *= 2 * np.pi

        mags = np.empty_like(angles)
        i_display = 5
        c = (pw[0] - 1) // 2
        for i, twist_angle in enumerate(angles):
            g = np.asarray([freq2pix(a, p)[1] for a, p in zip(amps, phases)])
            g = epslow + g * (epsg-epslow)
            
            if bilayer_mode == "copy":
                gratings = (g, g)
                depthss = (depths.copy(), depths.copy())
            elif bilayer_mode == "mirror":
                gratings = (g, np.flip(g, axis=0))
                depthss = (depths.copy(), np.flip(depths.copy(), axis=0))
            elif bilayer_mode == "free":
                gratings = np.split(g, 2, axis=0)
                depthss = np.split(depths.copy(), 2, axis=0)

            R, T, kxy = solve_multigrating(gratings, depthss, wavelength, twist_angle)
            mag = np.abs(T[1]).reshape(pw[0], pw[0])
            if i_display == i:
                mag_display = np.copy(mag)
            mags[i] = mag[c - 1, c + 1] / np.sum(mag)


        fom_percent = np.mean(mags*100)
        logging.debug(f"Grating sim got {fom_percent}% concentration.")


        if figpath is not None:
            bilayer_depth = np.sum(depthss, axis=1)
            gratings_picture = np.vstack((
                np.repeat( gratings[0], np.round(100 * depthss[0] / np.sum(depths)).astype(int), axis=0),
                np.repeat( gratings[1], np.round(100 * depthss[1] / np.sum(depths)).astype(int), axis=0),
            ))
            grating_summary_plot(figpath, angles, mags, kxy, mag_display, (angles[i_display], 100*mags[i_display]), fom_percent, gratings_picture, bilayer_depth)

        return mags


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Missing program name."
    if len(sys.argv) > 3:
        figpath = sys.argv[3]
    else:
        figpath = None
    program = sys.argv[1]
    assert len(sys.argv) > 2, "Missing input file."
    filepath = sys.argv[2]
    design = np.load(filepath)["bd"]
    design = design.reshape(16, 11).copy()
    main(program, design, figpath=figpath, bilayer_mode="free")


# Keever stuff
def __run__(program, design, angles, bilayer_mode, num_layers, figpath=None):
    logging.debug(f"{program=}{design.shape=}{angles=}{figpath=}")
    design = design.reshape(num_layers, 11).copy()
    angles = angles.copy()
    raw_fom = main(
        program=program,
        design=design,
        angles=angles,
        figpath=figpath,
        bilayer_mode=bilayer_mode,
    )
    fom = np.mean(raw_fom) #+ np.prod(raw_fom)
    return fom


def __requires__():
    return {"variables": ["program", "design", "angles", "bilayer_mode", "num_layers"]}
