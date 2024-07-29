'''
Study the diffraction efficiency of an half ellipsoid emerging from a substrate.
'''

from bast.draw import Drawing
from bast import Crystal, Expansion, Layer
import matplotlib.pyplot as plt
import numpy as np
from bast.misc import coords
import matplotlib.animation as animation
from multiprocessing import Pool
from itertools import product
from bast.alternative import incident
from functools import partial

pw = (3,1)

params = [0.4985506867145136, -0.029773681373547807, 0.14939078077380846, 0.49980704357804534, -0.7066760424782893]

def build_crystal(canvas, ta_deg=0):
  e1, e2 = Expansion(pw), Expansion(pw)
  e2.rotate(ta_deg)
  dlayer = 4 / canvas.shape[0]
  cl = Crystal.from_expansion(e1 + e2)
  device = []
  cl.add_layer("Sref", Layer.half_infinite(e1, "reflexion", 1.0), True)
  cl.add_layer("Strans", Layer.half_infinite(e2, "transmission", 1.0), True)
  for i, layer_eps in enumerate(canvas):
    name = f"l_{i}"
    if i < canvas.shape[0] // 2:
      ecur = e1
    else:
      ecur = e2
    if np.all(np.isclose(layer_eps, layer_eps[0])):
      cl.add_layer(name, Layer.uniform(ecur, layer_eps[0], dlayer), True)
    else:
      cl.add_layer(name, Layer.pixmap(ecur, layer_eps[:, np.newaxis], dlayer), True)
    device.append(name)
  cl.set_device(device, [True]*len(device))
  return cl


if False:
  orientations = np.linspace(-90, 90, 256)
  rt = []
  rt2 = []
  for o in orientations:
    d = Drawing(canvas_size, eps_bg)
    d.ellipse((0, params[1]),   (params[3],params[2]), np.deg2rad(o), eps_fg)
    cl, pic = build_crystal(d.canvas().T)
    cl.set_source(1.01, te=1,tm=1)
    cl.solve()
    (Rtot, R), (Ttot, T) = cl.poynting_flux_end(only_total=False)
    c = pw[0] // 2
    rt.append((T[c], Ttot))
    rt2.append((R[c], Rtot))
  rt = np.array(rt)
  rt2 = np.array(rt2)
  fig, (ax1, ax2) = plt.subplots(2, 1)
  ax1.plot(orientations, rt[:,1], color="r")
  ax1.plot(orientations, rt2[:,1], color="b")
  ax1.plot(orientations, rt2[:,1]+rt[:,1], color="k")
  d = Drawing(canvas_size, eps_bg)
  d.ellipse((0, params[1]),   (params[3],params[2]), params[4], eps_fg)
  pic = d.canvas()
  ax2.matshow(np.tile(pic.T,(1,5)), origin="lower", extent=[0,5,0,4], cmap="Blues")
  ax2.axis("equal")
  ax1.axvline(np.rad2deg(params[4]), color='k', ls=":")
  plt.tight_layout()
  plt.show()

if False:
  d = Drawing(canvas_size, eps_bg)
  d.ellipse((0, params[1]),   (params[3],params[2]), params[4], eps_fg)
  cl, pic = build_crystal(d.canvas().T)
  cl.set_source(1.01, te=1,tm=0)
  cl.solve()
  xres, yres = 1024, 128
  zres = 256
  xcells = 12
  x, y, z = coords(0, xcells, 0.0, 1.0, -0.1, cl.depth+3, (xres, yres, zres))

  E, H = cl.fields_volume(x, y, z)
  extent = [0, xcells, 0, cl.depth+3]
  im = plt.matshow(E[:, 0, :, 128].real, extent=extent, cmap="RdBu", origin="lower")
  extent = [0, xcells, 0, cl.depth]
  plt.contour(np.tile(pic.T, (1,xcells)), levels=[3], extent=extent)
  plt.axhline(4, color='k')
  plt.axis("equal")
  plt.colorbar(im)
  plt.show()


'''
Compute the fields for different twist angles and make a gif.
'''
if False:
  cl, pic = build_crystal(np.rad2deg(params[4]))
  fig, ax = plt.subplots()

  xres, yres = 512, 128
  zres = 128
  xcells = 12
  extent = [0, xcells, 0, 4+3]
  im = ax.matshow(pic, extent=extent, cmap="RdBu", origin="lower", vmin=-1, vmax=1)
  ax.set(xlim=[0, xcells], ylim=[0, 7], xlabel='X [um]', ylabel='Y [um]')
  ax.legend()
  ax.axis("equal")
  N = 128
  angles = np.linspace(0.01, 59.99, N)

  def update(frame):
    print(frame, N)
    cl, _ = build_crystal(np.rad2deg(params[4]), ta=angles[frame])
    cl.set_source(1.01, te=1,tm=0)
    cl.solve()
   
    x, y, z = coords(0, xcells, 0.0, 1.0, -0.1, cl.depth+3, (xres, yres, zres))

    E, H = cl.fields_volume(x, y, z)
    # for each frame, update the data stored on each artist.
    im.set_data(E[:, 0, :, yres//2].real)
    ax.set_title(str(frame))
    return (im)


  ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=200)
  ani.save("test.gif")


def draw_parallelogram(canvas, x0, y0, w, h, r):
  res = canvas.shape
  y = np.linspace(0, 4, res[0])
  x = x0 + np.linspace(0, 1, res[1])
  X, Y = np.meshgrid(x, y, indexing="xy")
  r = 90 - r
  slope = np.tan(np.deg2rad(r))
  for i in range(canvas.shape[0]):
    
    if y[i] <= y0:
      continue
    elif y[i] >= y0+h:
      break
    else:
      start = (y[i]) / slope
      end = start + w
      canvas[i, (x>=start) & (x < end)] = 4.0

def get_canvas_para(r, h, w):
  flip = r < 0
  r = abs(r)
  res = (128, 256)
  canvas = 2 * np.ones(res)
  yslab = 4
  y0 = (yslab-h) / 2
  draw_parallelogram(canvas, 0, y0, w, h, r)
  draw_parallelogram(canvas, 1, y0, w, h, r)
  draw_parallelogram(canvas, 2, y0, w, h, r)
  draw_parallelogram(canvas, -1,y0, w, h, r)

  if flip:
    return np.fliplr(canvas)
  else:
    return canvas

def worker(config):
  h, o = config
  pic = get_canvas_para(o, h, 0.5)
  cl = build_crystal(pic, ta_deg=30)
  cl.set_source(1.01, te=0,tm=1)
  incident_fields = incident(
      cl.pw,
      cl.source.te,
      cl.source.tm,
      k_vector=(cl.kp[0], cl.kp[1], cl.kzi),
  )
  cl.solve()
  Wref = cl.layers["Sref"].W
  iWref = np.linalg.inv(Wref)
  c1p = iWref @ incident_fields
  c2p = np.split(cl.Stot[1, 0] @ c1p, 2)[1]
  (Rtot, R), (Ttot, T) = cl.poynting_flux_end(only_total=False)
  c = pw[0] // 2
  T = T.reshape(pw[0], pw[0])
  return (Rtot, *c2p.flatten())

if True:
  no, nh = 128, 128
  orientations = np.linspace(-40, 40, no)
  heights = np.linspace(0.5, 4.0, nh)
  rt = []
  configs = list(product(heights, orientations))
  with Pool(16) as p:
    rt = p.map(worker, configs)
  mag = np.abs(np.array(rt).reshape(nh, no, 10))
  phase = np.angle(np.array(rt).reshape(nh, no, 10))

  fig, axs = plt.subplots(3, 3, figsize=(6,6), dpi=300)
  axs = axs.flatten()
  #names = ["+1/-1", "0/0", "-1/+1", "tot"]
  magt = np.sum(mag[..., 1:], axis=-1)
  mag /= magt[..., np.newaxis]
  for i in range(9):
    im = axs[i].matshow(mag[..., i + 1], extent=[0, 1, 0,1], vmin=0)
    plt.colorbar(im)
    axs[i].set_xticks([0,1])
    axs[i].set_xticklabels(map(str, [np.min(orientations), np.max(orientations)]))
    axs[i].set_yticks([0,1])
    axs[i].set_yticklabels(map(str, [np.min(heights), np.max(heights)]))
  plt.tight_layout()
  fig.savefig("OrientationInfluence.png")

  fig, axs = plt.subplots(3, 3, figsize=(6,6), dpi=300)
  axs = axs.flatten()
  #names = ["+1/-1", "0/0", "-1/+1", "tot"]
  for i in range(9):
    alpha = np.clip(mag[..., i+1], 0, 1)
    im = axs[i].matshow(phase[..., i + 1], extent=[0, 1, 0,1], cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axs[i].set_xticks([0,1])
    axs[i].set_xticklabels(["-90", "90"])
    axs[i].set_yticks([0,1])
    axs[i].set_yticklabels(["0.1", "4.0"])
    plt.colorbar(im)
  plt.tight_layout()
  fig.savefig("OrientationInfluencePhase.png")

# Viz the parallelograms
if False:
  no, nh = 16, 16
  orientations = np.linspace(-50, 50, no)
  heights = np.linspace(0.5, 4.0, nh)
  fig, axs = plt.subplots(no, nh, dpi=300, figsize=(4,6), gridspec_kw = {'wspace':0, 'hspace':0})
  axs = axs.flatten()
  for (o, h), ax in zip(product(orientations, heights), axs):
    pic = get_canvas_para(o, h, 0.5)
    ax.matshow(pic, cmap="Blues", extent=[0, 1, 0, 4])
    ax.axis("off")
  fig.savefig("Parallelograms.png")

