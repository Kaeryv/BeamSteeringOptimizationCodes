import matplotlib.pyplot as plt
import numpy as np

import parameterization as prm

plt.rcParams.update({"font.size": 20})


def grating_side_picture(gratings, layers_depth, minimal_feature_size, ppm=24, pic_height=6):
    bilayer_depth = np.sum(layers_depth, axis=1)
    gratings_picture = [
        np.repeat(
            [prm.grating_filtering(g, width=minimal_feature_size) for g in grating],
            np.ceil(ppm * slab_layers_depth).astype(int),
            axis=0,
        )
        for slab_layers_depth, grating in zip(layers_depth, gratings)
    ]
    gratings_picture = np.vstack(gratings_picture).astype(float)
    target_height_px = ppm * pic_height
    if gratings_picture.shape[0] > target_height_px:
        print("[ERROR]")
    elif gratings_picture.shape[0] < target_height_px:
        missing = target_height_px - gratings_picture.shape[0]
        if missing % 2 == 0:
            pad = np.ones((missing//2, gratings_picture.shape[1]))
            gratings_picture = np.vstack((pad, gratings_picture, pad))
        else:
            padu = np.ones((missing//2, gratings_picture.shape[1]))
            padd = np.ones((missing//2+1, gratings_picture.shape[1]))
            gratings_picture = np.vstack((padu, gratings_picture, padd))
    #print(gratings_picture.shape)
    return gratings_picture, bilayer_depth


def plot_angle_magnitude(ax, angles, magnitudes, style={}):
    fom_percent = np.mean(magnitudes) * 100
    ax.plot(angles, magnitudes * 100, **style)
    ax.set_xlim(0, 61)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Twist angle $\\alpha$ [deg]")
    ax.set_ylabel("(1,-1) magnitude [%]")
    ax.axhline(fom_percent, color="k")
    ax.set_title(f"avg. {fom_percent:0.1f}")


def plot_highlight(ax, angle, magnitude):
    xydisp = (angle, 100 * magnitude)
    ax.plot(*xydisp, "ko", markerfacecolor="none", markersize=20)

def plot_grid(ax, kxy, mags):
    ax.scatter(*kxy, s=100, marker="o", facecolors="none", edgecolors="gray")
    ax.scatter(*kxy, s=mags / np.max(mags) * 100, c="r")
    ax.set_xlabel("$k_x$", labelpad=0)
    ax.set_ylabel("$k_y$", labelpad=-10)
    ax.set_title("$\\alpha=60$")

def grating_summary_plot(
    figpath,
    angles,
    magnitudes,
    kxy,
    mag_display,
    gratings_picture,
    layers_depth,
    tiles=4,
):
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 7))
    ((ax1, ax2), (ax3, ax4)) = axs
    
    plot_highlight(ax1, angles[5], magnitudes[5])

    plot_grid(ax2, kxy, mag_display)
    extent = [tiles - 2, tiles - 1, 0, 6]
    ax3.matshow(
        np.tile(gratings_picture, (1, 1)),
        cmap="Blues",
        alpha=0.7,
        extent=extent,
        aspect=1 / 3,
    )
    extent = [0, tiles, 0, 6]
    ax3.matshow(
        np.tile(gratings_picture, (1, tiles)),
        cmap="Blues",
        alpha=0.3,
        extent=extent,
        aspect=1 / 3,
    )
    ax3.axhline(layers_depth[1], c="r", ls="-.")
    ax3.axvline(2, c="k", ls=":")
    ax3.axvline(3, c="k", ls=":")
    ax3.xaxis.tick_bottom()
    ax3.set_xlabel("x [um]")
    ax3.set_ylabel("z [um]")
    ax3.axis("equal")

    ax4.axis("off")

    plt.tight_layout()
    return fig, axs
