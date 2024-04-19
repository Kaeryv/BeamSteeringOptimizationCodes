import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 20})


def grating_summary_plot(figpath, angles, magnitudes, kxy, mag_display, xydisp, fom_percent, gratings_picture, layers_depth, tiles=4, profile=None):
    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(7.5,7))
    ax1.plot(angles, magnitudes * 100, "r.-")
    ax2.scatter(*kxy, s=100, marker="o", facecolors="none", edgecolors="gray")
    ax2.scatter(*kxy, s=mag_display / np.max(mag_display) * 100, c="r")
    ax2.set_xlabel("$k_x$",labelpad=0)
    ax2.set_ylabel("$k_y$",labelpad=-10)
    ax2.set_title("$\\alpha=60$")
    ax1.set_xlabel("Twist angle $\\alpha$ [deg]")
    ax1.set_xlim(0, 61)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("(1,-1) magnitude [%]")
    ax1.plot(*xydisp, 'ko',  markerfacecolor="none", markersize=20)
    ax1.set_title(f"avg. {fom_percent:0.1f}") # geoavg. {(np.prod(mags))**(1/len(mags)):0.3f}
    ax1.axhline(fom_percent, color="k")
    extent = [tiles-2,tiles-1,0, np.sum(layers_depth)]
    ax3.matshow(np.tile(gratings_picture, (1, 1)), cmap="Blues", alpha=0.7, extent=extent, aspect=1/3)
    extent = [0,tiles,0, np.sum(layers_depth)]
    ax3.matshow(np.tile(gratings_picture, (1,tiles)), cmap="Blues", alpha=0.3, extent=extent, aspect=1/3)
    ax3.axhline(layers_depth[1], c="r", ls="-.")
    ax3.axvline(2, c="k", ls=":")
    ax3.axvline(3, c="k", ls=":")
    ax3.xaxis.tick_bottom()
    ax3.set_xlabel("x [um]")
    ax3.set_ylabel("z [um]")

    if profile is not None:
        ax4.plot(profile[:,0], profile[:,1], 'ro')
        ax4.set_xlabel("Number of simulations")
        ax4.set_ylabel("Figure of merit.")
    else:
        ax4.axis('off')
    
    plt.tight_layout()
    fig.savefig(figpath, transparent=True)
    del fig
