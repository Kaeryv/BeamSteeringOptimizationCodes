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

pw = (5,1)
pwt = (pw[0],pw[0])

def draw_parallelogram(canvas, x0, y0, w, h, r, epsilon=4):
  res = canvas.shape
  y = np.linspace(0, 4, res[0])
  x = x0 + np.linspace(0, 1, res[1])
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
      canvas[i, (x>=start) & (x < end)] = epsilon

def get_canvas_para(o, h, w, resolution = (128, 256), epsilon=4, epsilon_bg=2):
  flip = o < 0
  o = abs(o)
  
  canvas = epsilon_bg * np.ones(resolution)
  yslab = 4
  y0 = (yslab-h) / 2
  draw_parallelogram(canvas, -0.25, y0, w, h, o, epsilon)
  draw_parallelogram(canvas, -1.25, y0, w, h, o, epsilon)
  draw_parallelogram(canvas, 0.75, y0, w, h, o, epsilon)
  draw_parallelogram(canvas, 1.75,y0, w, h, o, epsilon)

  if flip:
    return np.fliplr(canvas)
  else:
    return canvas

def build_crystal(canvas, ta_deg=0, height=4):
  e1, e2 = Expansion(pw), Expansion(pw)
  e2.rotate(ta_deg)
  dlayer = height / canvas.shape[0]
  cl = Crystal.from_expansion(e1 + e2)
  cl.add_layer("Sref", Layer.half_infinite(e1, "reflexion", 1.0), True)
  cl.add_layer("Strans", Layer.half_infinite(e2, "transmission", 1.0), True)
  
  device = []
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
  fig, ax = plt.subplots(figsize=(6,4), dpi=300)
  #d = Drawing(canvas_size, eps_bg)
  #d.ellipse((0, params[1]),   (params[3],params[2]), params[4], eps_fg)
  pic = get_canvas_para(-20.0, 3.0, 0.5, epsilon=4, epsilon_bg=2, resolution=(128,256))
  ta_deg=20
  cl = build_crystal(pic, ta_deg=ta_deg)
  cl.set_source(1.01, te=1,tm=0)
  cl.solve()
  xres, yres = 2, 128
  zres = 512
  xcells = 8
  x, y, z = coords(0, xcells, 0.5, 0.5, -0.1, cl.depth+3, (xres, yres, zres))

  E, H = cl.fields_volume(x, y, z)
  extent = [0, xcells, 0, cl.depth+3]
  im = ax.matshow(np.real(E[:, 0, :, xres//2]), extent=extent, cmap="RdBu", origin="lower")
  extent = [0, xcells, 0, cl.depth/2]
  ax.contourf((np.tile(pic[:64, :], (1,xcells))), levels=[3,4], extent=extent, colors=['k'], alpha=0.3)
  extent = [0, xcells/ np.sin(np.deg2rad(90-ta_deg)), cl.depth/2, cl.depth]
  ax.contourf((np.tile(pic[64:, :], (1,xcells))), levels=[3,4], extent=extent, colors=['k'], alpha=0.3)
  ax.axhline(4, color='k')
  ax.axhline(2, color='k', ls=":")
  ax.axis("equal")
  ax.set_xlabel("X [µm]")
  ax.set_ylabel("Z [µm]")
  ax.set_xlim(0, xcells)
  plt.colorbar(im)
  fig.savefig('Fields.png')

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


polars = {"te": (1,0), "tm": (0,1), "tem": (1,1)}
polar = "tem"
def worker(config):
  h, o = config
  pic = get_canvas_para(o, h, 0.5, epsilon=4, epsilon_bg=2, resolution=(128,256))
  #pic = get_canvas_para(o, 3.0, h, epsilon=4, epsilon_bg=2, resolution=(128,256))
  cl = build_crystal(pic, ta_deg=20) #, height=h
  cl.set_source(1.01, te=polars[polar][0],tm=polars[polar][1])
  cl.solve()
  Wref = cl.layers["Sref"].W
  Wtrans = cl.layers["Strans"].W
  incident_fields = incident(
      cl.pw,
      cl.source.te,
      cl.source.tm,
      k_vector=(cl.kp[0], cl.kp[1], cl.kzi),
  )
  c1p = np.linalg.inv(Wref) @ incident_fields
  sx, sy = np.split(Wtrans @ cl.Stot[1, 0] @ c1p, 2)
  kx, ky, kz = cl.expansion.k_vectors(cl.kp, 1.01, 1)
  fac = 2*np.pi/1.01 * kz.real / cl.kzi
  #(Rtot, R), (Ttot, T) = cl.poynting_flux_end(only_total=False)
  #T = T.reshape(pw[0], pw[0])
  return sy.flatten() * np.sqrt(fac.real)

import os
if True:
  no, nh = 128, 128
  orientations = np.linspace(-40, 40, no)
  #heights = np.linspace(1.5, 4.0, nh)
  heights = np.linspace(0.05, 0.95, nh)
  rt = []
  configs = list(product(heights, orientations))
  filename = "understand_twisted.npy"
  num_orders = pwt[0] * pwt[1]
  if not os.path.isfile(filename):
    print("File does not exist. Computing.")
    with Pool(16) as p:
      rt = p.map(worker, configs)
    rt = np.array(rt).reshape(nh, no, num_orders)
    np.save(filename, rt)
  else:
    rt = np.load(filename)
  mag = np.abs(rt)**2
  phase = np.angle(rt)

  def setlims(ax):
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels(["-40", "0", "40"])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels([np.max(heights),(np.max(heights)+np.min(heights))/2, np.min(heights)])
    ax.xaxis.tick_bottom()
  
  def setup_axes(axs):
    for ax in axs[-1, :]:
      ax.set_xlabel("Slant angle [deg]")
    for ax in axs[:, 0]:
      ax.set_ylabel("Para. Width [um]")
    
    for j, ax in np.ndenumerate(axs):
      ax.set_title(f"{[i-2 for i in j]}")

  fig, axs = plt.subplots(pwt[0], pwt[1], figsize=(12,12), dpi=300)
  setup_axes(axs)
  axs = axs.flatten()
  for i in range(num_orders):
    im = axs[i].matshow(mag[..., i], extent=[0, 1, 0,1], vmin=0, vmax=1)
    setlims(axs[i])
  fig.subplots_adjust(right=0.9, left=0.08, top=0.95, bottom=0.08)
  cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
  fig.colorbar(im, cax=cbar_ax)
  fig.savefig(f"OrientationInfluence.{polar}.png")

  fig, axs = plt.subplots(pwt[0], pwt[1], figsize=(12,12), dpi=300)
  setup_axes(axs)
  axs = axs.flatten()
  for i in range(num_orders):
    alpha = np.clip(mag[..., i], 0, 1)
    im = axs[i].matshow(phase[..., i], extent=[0, 1, 0,1], cmap="twilight", vmin=-np.pi, vmax=np.pi)
    setlims(axs[i])
    axs[i].contour(mag[..., i], extent=[0, 1, 1,0], levels=[0.6*np.max(mag[..., i])], colors=["g"])
  fig.subplots_adjust(right=0.9, left=0.08, top=0.95, bottom=0.08)
  cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
  fig.colorbar(im, cax=cbar_ax)
  fig.savefig(f"OrientationInfluencePhase.{polar}.png")

# Viz the parallelograms
if False:
  no, nh = 16, 16
  orientations = np.linspace(-10, 10, no)
  heights = np.linspace(0.1, 0.9, nh)
  fig, axs = plt.subplots(no, nh, dpi=300, figsize=(4,6), gridspec_kw = {'wspace':0, 'hspace':0})
  axs = axs.flatten()
  for (o, h), ax in zip(product(orientations, heights), axs):
    pic = get_canvas_para(o, h, 0.5, resolution=(128, 256))
    ax.matshow(pic, cmap="Blues", extent=[0, 1, 0, 4])
    ax.axis("off")
  fig.savefig("Parallelograms.png")

