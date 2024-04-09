from bast.layer import Layer, stack_layers
from bast.extension import ExtendedLayer as EL
from bast.expansion import Expansion
from bast.crystal import Crystal

import sys

sys.path.append(".")

from user.commons import freq2pix

import numpy as np
import matplotlib.pyplot as plt
import logging

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
    # Build the upper crystal
    crystal_up = build_substack(e1, X1, depths)
    crystal_do = build_substack(e2, X2, depths)

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


def main(program, design=None, angles=None, figpath=None, bilayer_mode="copy"):
    epsg = 4

    if program == "showme":
        N = 8
        amps = np.random.rand(N, 5)
        phases = np.random.rand(N, 5) * 2 * np.pi
        g = np.asarray([freq2pix(a, p)[1] for a, p in zip(amps, phases)])
        g = (1 + g) * epsg
        depths = np.random.rand(N)
        R, T, kxy = solve_multigrating((g, g), depths, 0.6, 3)
        mag = np.abs(T[1])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        gratings_depth = np.repeat(
            g, np.round(100 * depths / np.sum(depths)).astype(int), axis=0
        )
        ax1.matshow(gratings_depth, cmap="Blues")
        ax2.scatter(*kxy, s=100, marker="o", facecolors="none", edgecolors="gray")
        ax2.scatter(*kxy, s=mag / np.max(mag) * 100, c="r")
        ax1.axis("equal")
        fig.savefig("figs/multigrating.png")

    elif program == "angle_batch":
        if angles is None:
            angles = np.linspace(0, 60, 30)
        else:
            angles = np.linspace(angles[0], angles[1], angles[2])

        amps, phases, depths = np.split(design, [5, 10], axis=1)
        depths = np.squeeze(depths)
        phases *= 2 * np.pi

        mags = np.empty_like(angles)
        c = (pw[0] - 1) // 2
        for i, twist_angle in enumerate(angles):
            g = np.asarray([freq2pix(a, p)[1] for a, p in zip(amps, phases)])
            g = (1 + g) * epsg
            if bilayer_mode == "copy":
                gratings = (g, g)
            elif bilayer_mode == "mirror":
                gratings = (g, np.flip(g, axis=0))
            elif bilayer_mode == "free":
                gratings = np.split(g, 2, axis=0)
            R, T, kxy = solve_multigrating(gratings, depths, 0.48, twist_angle)
            mag = np.abs(T[1]).reshape(5, 5)
            mags[i] = mag[c - 1, c + 1] / np.sum(mag)

        if figpath is not None:
            fig, ((ax1, ax2), (ax3,_)) = plt.subplots(2, 2)
            ax1.plot(angles, mags * 100, "r.-")
            ax2.scatter(*kxy, s=100, marker="o", facecolors="none", edgecolors="gray")
            ax2.scatter(*kxy, s=mag / np.max(mag) * 100, c="r")
            ax1.set_xlabel("Twist angle [deg]")
            ax1.set_ylabel("(-1,1) magnitude [%]")
            ax1.set_title(f"{np.mean(mags)=:0.3f}, {np.prod(mags)=:0.3f}")
            gratings_depth = np.repeat(
                g, np.round(100 * depths / np.sum(depths)).astype(int), axis=0
            )
            ax3.matshow(gratings_depth, cmap="Blues")
            fig.savefig(figpath)

        return mags


#if __name__ == "__main__":
#    assert len(sys.argv) > 1, "Missing program name."
#    if len(sys.argv) > 3:
#        figpath = sys.argv[3]
#    else:
#        figpath = None
#    program = sys.argv[1]
#    # assert len(sys.argv) > 2, "Missing input file."
#    N = 8
#    design = np.random.rand(N, 11)
#    main(program, design, figpath=figpath)


# Keever stuff
def __run__(program, design, angles, bilayer_mode, figpath=None):
    logging.debug(f"{program=}{design.shape=}{angles=}{figpath=}")
    design = design.reshape(8, 11).copy()
    angles = angles.copy()
    raw_fom = main(
        program=program,
        design=design,
        angles=angles,
        figpath=figpath,
        bilayer_mode=bilayer_mode,
    )
    fom = np.mean(raw_fom) + np.prod(raw_fom)
    return fom


def __requires__():
    return {"variables": ["program", "design", "angles", "bilayer_mode"]}
