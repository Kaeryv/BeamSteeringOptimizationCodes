'''
Study the diffraction efficiency of an half ellipsoid emerging from a substrate.
'''

import bast
from bast.draw import Drawing
from bast.crystal import Crystal
import matplotlib.pyplot as plt
import numpy as np
from bast.misc import coords

canvas_size = 128, 16
pw = (5,1)

params = [0.4985506867145136, -0.029773681373547807, 0.14939078077380846, 0.49980704357804534, -0.7066760424782893]

def build_crystal(orientation_deg, eps_bg=2, eps_fg=4):
  d = Drawing(canvas_size, eps_bg)
  d.ellipse((0, params[1]),   (params[3],params[2]), np.deg2rad(orientation_deg), eps_fg)#params1[3],params1[2]

  #d.rectangle((0, -0.25), (1,0.5), eps_fg)
  #d.plot()
  dlayer = 4 / canvas_size[1]
  cl = Crystal(pw, epse=1.0, epsi=1.0)
  device = []
  for i, layer_eps in enumerate(d.canvas().T):
    name = f"l_{i}"
    if np.all(np.isclose(layer_eps, layer_eps[0])):
      cl.add_layer_uniform(name, layer_eps[0], dlayer)
    else:
      cl.add_layer_pixmap(name, layer_eps[:, np.newaxis], dlayer)
    device.append(name)
  cl.set_device(device, [True]*len(device))
  return cl, d.canvas()

if True:
  orientations = np.linspace(-90, 90, 256)
  rt = []
  rt2 = []
  for o in orientations:
    cl, pic = build_crystal(o)
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
  o = np.rad2deg(params[4])
  _, pic = build_crystal(o)
  ax2.matshow(np.tile(pic.T,(1,5)), origin="lower", extent=[0,5,0,4], cmap="Blues")
  ax2.axis("equal")
  ax1.axvline(np.rad2deg(params[4]), color='k', ls=":")
  plt.tight_layout()

  plt.show()

if False:
  cl, pic = build_crystal(np.rad2deg(params[4]))
  cl.set_source(1.01, te=1,tm=0)
  cl.solve()
  xyres = 256
  zres = 256
  x, y, z = coords(0, 1, 0.0, 1.0, -0.1, cl.depth+2, (xyres, xyres, zres))

  E, H = cl.fields_volume(x, y, z)
  print(E.shape)
  extent = [0, 4, 0, 6]
  im = plt.matshow(np.tile(E[:, 0, :, 128].real, (1, 4)), extent=extent, cmap="RdBu", origin="lower")
  extent = [0, 4, 0, 4]
  plt.contour(np.tile(pic.T, (1,4)), levels=[3], extent=extent)
  plt.axhline(4, color='k')
  plt.axis("equal")
  plt.colorbar(im)
  plt.show()







