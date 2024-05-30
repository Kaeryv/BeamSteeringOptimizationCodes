import numpy as np
from user.commons import freq2pix
from user.ellipse import draw_ellipse 
from PIL import Image, ImageDraw 
import matplotlib.pyplot as plt


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

def fftlike(X, elow, ehigh, bilayer_mode, num_layers=12, harmonics=None):
    #X = X.reshape(-1, 2*len(harmonics)+1)
    X = X.flatten()
    amps, phases, depths = np.split(X, [num_layers*len(harmonics), 2*num_layers*len(harmonics)], axis=0)
    amps = amps.reshape(num_layers, len(harmonics))
    phases = phases.reshape(num_layers, len(harmonics))
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

def ellipsis(X, elow, ehigh, bilayer_mode, num_items=3, num_layers=16):
    xys, axes, angles = np.split(X, [num_items*2, 2*2*num_items], axis=0)
    xys = xys.reshape(num_items, 2)
    axes = axes.reshape(num_items, 2)
    angles = np.squeeze(angles)

    w, h = 256, 256
    g = Image.new("L", (w, h))
    for (x,y), (a, b), alpha in zip(xys, axes, angles):
        draw_ellipse(g, (x,y,a,b,alpha))
    g = g.resize((256, 16), Image.NEAREST)
    g = np.asarray(g)
    g = elow + g * (ehigh-elow)
    depths = np.ones(num_layers) * 4 / num_layers
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
