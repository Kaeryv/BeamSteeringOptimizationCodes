import numpy as np
from user.commons import freq2pix


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

def fftlike(X, elow, ehigh, bilayer_mode, harmonics=None):
    amps, phases, depths = np.split(X, [len(harmonics), 2*len(harmonics)], axis=1)
    depths = np.squeeze(depths)
    phases *= 2 * np.pi

    g = np.asarray([freq2pix(a, p, harmonics=harmonics)[1] for a, p in zip(amps, phases)])
    g = elow + g * (ehigh-elow)

    if bilayer_mode == "copy":
        gratings = (g, g)
        depthss = (depths.copy(), depths.copy())
    elif bilayer_mode == "mirror":
        gratings = (g, np.flip(g, axis=0))
        depthss = (depths.copy(), np.flip(depths.copy(), axis=0))
    elif bilayer_mode == "free":
        gratings = np.split(g, 2, axis=0)
        depthss = np.split(depths.copy(), 2, axis=0)

    return gratings, depthss

def placeblocks(X, elow, ehigh, bilayer_mode, num_blocks):
    centers, widths, depths = np.split(X, [num_blocks, 2*num_blocks], axis=1)
    depths = np.squeeze(depths)

    def coords2pix(cs, ws):
        x = np.linspace(0, 1, 256)
        canvas = np.zeros_like(x)
        for c, w in zip(cs, ws):
            mask = abs(x-c) < w/2
            canvas[mask] = 1
        return canvas

    g = np.asarray([coords2pix(c, w) for c, w in zip(centers, widths)])
    g = elow + g * (ehigh-elow)

    if bilayer_mode == "copy":
        gratings = (g, g)
        depthss = (depths.copy(), depths.copy())
    elif bilayer_mode == "mirror":
        gratings = (g, np.flip(g, axis=0))
        depthss = (depths.copy(), np.flip(depths.copy(), axis=0))
    elif bilayer_mode == "free":
        gratings = np.split(g, 2, axis=0)
        depthss = np.split(depths.copy(), 2, axis=0)

    return gratings, depthss
    