import numpy as np
from user.commons import freq2pix
from user.ellipse import draw_ellipse, draw_ellipse_old
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

def grating_filtering(X, width=1, elow=2, ehigh=4):

    bx = X > (elow + ehigh) / 2
    bx = binary_erosion(bx, width)
    bx = binary_dilation(bx, width)
    bx = np.logical_not(bx)
    bx = binary_erosion(bx, width)
    bx = binary_dilation(bx, width)
    bx = np.logical_not(bx)

    return (ehigh - elow) * bx.astype(float) + elow

def apply_bilayer_mode(raw_gratings, raw_depths, mode):
    '''
    Takes as input the raw gratings and split them in the two bilayers
    following the specified 'mode'.
    copy: the given bilayer is copy-translated from the first to the second
    layer.
    mirror: Same as copy but using mirror symmetry around the twist plane.
    free: The given gratings are split in two equal-sized groups. Top and bottom
    crystals.
    '''
    if mode == "copy":
        return (raw_gratings, raw_gratings), (raw_depths.copy(), raw_depths.copy())
    elif mode == "mirror":
        return (raw_gratings, np.flip(raw_gratings, axis=0)), (raw_depths.copy(), np.flip(raw_depths.copy(), axis=0))
    elif mode == "free":
        return np.split(raw_gratings.copy(), 2, axis=0), np.split(raw_depths.copy(), 2, axis=0)
    else:
        print("Unsupported bilayer mode.")
        exit(1)

def fftlike(X, elow, ehigh, bilayer_mode, num_layers=12, harmonics=None, **kwargs):
    X = X.flatten()
    amps, phases, depths = np.split(X, [num_layers*len(harmonics), 2*num_layers*len(harmonics)], axis=0)
    amps = amps.reshape(num_layers, len(harmonics))
    phases = phases.reshape(num_layers, len(harmonics))
    depths = np.squeeze(depths)
    phases *= 2 * np.pi

    g = np.asarray([freq2pix(a, p, harmonics=harmonics)[1] for a, p in zip(amps, phases)])
    g = elow + g * (ehigh-elow)
    return apply_bilayer_mode(g, depths, bilayer_mode)

def placeblocks(X, elow, ehigh, bilayer_mode, num_blocks, **kwargs):
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
    return apply_bilayer_mode(g, depths, bilayer_mode)

def ellipsis(X, elow, ehigh, bilayer_mode, num_items=3, depth=4, num_layers=16, **kwargs):
    xys, axes, angles = np.split(X, [num_items*2, 2*2*num_items], axis=0)
    xys = xys.reshape(num_items, 2)
    axes = axes.reshape(num_items, 2)
    angles = np.asarray(angles).flatten()

    w, h = 256, num_layers
    canvas = np.zeros((h, w))
    def _draw(x, y, a, b, alpha):
        draw_ellipse(canvas, (x, y, a, b, alpha))
    
    for (x,y), (a, b), alpha in zip(xys, axes, angles):
        # Tiling 9 patterns to account for periodicity
        _draw(x,   y,   a, b, alpha)
        _draw(x-1, y,   a, b, alpha)
        _draw(x+1, y,   a, b, alpha)
        #_draw(x,   y+1, a, b, alpha)
        #_draw(x-1, y+1, a, b, alpha)
        #_draw(x+1, y+1, a, b, alpha)
        #_draw(x,   y-1, a, b, alpha)
        #_draw(x-1, y-1, a, b, alpha)
        #_draw(x+1, y-1, a, b, alpha)

    canvas = elow + canvas * (ehigh-elow)
    depths = np.ones(num_layers) * depth / num_layers
    return apply_bilayer_mode(canvas, depths, bilayer_mode)

def ellipsismm(X, elow, ehigh, bilayer_mode, num_items=3, depth=4, num_layers=16,  **kwargs):
    print(X.shape)
    xys, axes, angles, materials = np.split(X, [num_items*2, 2*2*num_items, 5*num_items], axis=0)
    xys = xys.reshape(num_items, 2)
    axes = axes.reshape(num_items, 2)
    angles = np.asarray(angles).flatten()
    materials = np.asarray(materials).flatten().astype(int)

    w, h = 256, num_layers
    proto_canvas = np.zeros((h, w))
    canvas = np.ones((h, w))
    def _draw(x, y, a, b, alpha):
        draw_ellipse(proto_canvas, (x, y, a, b, alpha))
    
    for (x,y), (a, b), alpha, mat in zip(xys, axes, angles, materials):
        # Tiling 9 patterns to account for periodicity
        proto_canvas = np.zeros((h, w))
        _draw(x,   y,   a, b, alpha)
        _draw(x-1, y,   a, b, alpha)
        _draw(x+1, y,   a, b, alpha)
        canvas[proto_canvas > 0.5] = kwargs["materials"][mat]
        #_draw(x,   y+1, a, b, alpha)
        #_draw(x-1, y+1, a, b, alpha)
        #_draw(x+1, y+1, a, b, alpha)
        #_draw(x,   y-1, a, b, alpha)
        #_draw(x-1, y-1, a, b, alpha)
        #_draw(x+1, y-1, a, b, alpha)
    depths = np.ones(num_layers) * depth / num_layers
    return apply_bilayer_mode(canvas, depths, bilayer_mode)

def ellipsisold(X, elow, ehigh, bilayer_mode, num_items=3, depth=4, num_layers=16, **kwargs):
    xys, axes, angles = np.split(X, [num_items*2, 2*2*num_items], axis=0)
    xys = xys.reshape(num_items, 2)
    axes = axes.reshape(num_items, 2)
    angles = np.asarray(angles).flatten()
    #xys -= 0.5
    #axes /= 2
    #angles = np.rad2deg(angles)

    w, h = 256, 256
    g = Image.new("L", (w, h))
    
    for (x,y), (a, b), alpha in zip(xys, axes, angles):
        draw_ellipse_old(g, (x,y,a,b,alpha))

    g = g.resize((256, 16), Image.NEAREST)
    g = np.asarray(g).astype(float) / 255

    g = elow + g * (ehigh-elow)
    depths = np.ones(num_layers) * depth / num_layers
    return apply_bilayer_mode(g, depths, bilayer_mode)

def rawimg(X, elow, ehigh, bilayer_mode, num_layers=16, **kwargs):
    X2, depth = X
    depths = np.ones(num_layers) * depth / num_layers # 2-4
    return apply_bilayer_mode(X2, depths, bilayer_mode)
