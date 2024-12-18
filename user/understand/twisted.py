import sys
sys.path.append(".")
from bast import Crystal, Expansion, Layer
import matplotlib.pyplot as plt
import numpy as np
from bast.misc import coords
from multiprocessing import Pool
from itertools import product
import os
from bast.draw import Drawing
from functools import partial
from types import SimpleNamespace
import user.parameterization as prm

pw = (5,1)
pwt = (pw[0],pw[0])
program = sys.argv[1]

def pic_from_npz():
  design = np.load(sys.argv[5])["bd"]
  sim_args = SimpleNamespace(
      elow=2.0,
      ehigh=4.0,
      wavelength=1.01,
      bilayer_mode="free",
      num_layers=16,
      parameterization="ellipsis",
      target_order=(-1, +1),
      parameterization_args={"num_items": 12 , "num_layers": 16,"depth": 3.2,},#"materials": [2.0, 3.0, 4.0 ]},
  )
  gratings, depthss = getattr(prm, sim_args.parameterization)(
      design,
      sim_args.elow,
      sim_args.ehigh,
      sim_args.bilayer_mode,
      **sim_args.parameterization_args,
  )
  return np.vstack(gratings)


def parallelogram_inside_slab(o, h, w, eps_parallelogram, eps_background=2, hslab=4, resolution = (256, 256), interlayer=0):
  d = Drawing(resolution, eps_background, lattice=np.array([[1,0],[0, hslab]]))
  for x_offset in [ -1,  0, 1]:
    d.parallelogram((x_offset, 0), (w, h), o, eps_parallelogram)
  
  if interlayer > 0:
    d.rectangle((0,0), (1, interlayer), 2.0)
  return d.canvas().T

dbuffer = 0.99
def build_crystal(canvas, ta_rad=0, height=4, fields=False, onlyfirst=False):
  e1, e2 = Expansion(pw), Expansion(pw)
  e2.rotate(ta_rad/2)
  e1.rotate(-ta_rad/2)
  dlayer = height / canvas.shape[0]
  epse = 2 if onlyfirst else 1 # Substrate homogenous with background
  cl = Crystal.from_expansion(e1 + e2, epse=epse)
  cl.add_layer("Sref", Layer.half_infinite(e1, "reflexion", 1.0), True)
  cl.add_layer("Strans", Layer.half_infinite(e2, "transmission", epse), True)
  
  device = []
  for i, layer_eps in enumerate(canvas):
    name = f"l_{i}"
    if i < canvas.shape[0] // 2:
      ecur = e1
    else:
      ecur = e2
      if onlyfirst:
        cl.add_layer(name, Layer.uniform(ecur, 2, dlayer), True)
        device.append(name)
        continue
    if np.all(np.isclose(layer_eps, layer_eps[0])): # Implement pixmap_or_uniform
      cl.add_layer(name, Layer.uniform(ecur, layer_eps[0], dlayer), True)
    else:
      cl.add_layer(name, Layer.pixmap(ecur, layer_eps[:, np.newaxis], dlayer), True)
    device.append(name)
  if fields:
    cl.add_layer("Sbuffer", Layer.uniform(e1, 1.0, dbuffer), True)
    cl.add_layer("Sbuffer2", Layer.uniform(e2, epse, dbuffer), True)
    device.extend(["Sbuffer2"]*5)
    device.insert(0, "Sbuffer")
    device.insert(0, "Sbuffer")
  cl.set_device(device, [fields]*len(device))
  return cl

polars = {"te": (1,0), "tm": (0,1), "tem": (1,1)}
polar = "te"
hslab = 3.2

onlyfirst = sys.argv[2] == "first"
plane = sys.argv[3]
struct = sys.argv[4]

if program == "fields":
  if struct == "opt":
    pic = pic_from_npz()
    hslab = 3.2
    ta_deg = float(sys.argv[6])
  else:
    pic = parallelogram_inside_slab(np.deg2rad(-22.5), 2.7, 0.4, eps_parallelogram=4, eps_background=2, hslab=hslab)
    ta_deg = float(sys.argv[5])
  ta_rad = np.deg2rad(ta_deg)
  xres, yres = 128, 256
  zres = 256
  xcells = 12
  dafter = 6.9
  dair = 2 *  dbuffer
  dbefore = -2.1
  filename = "E.npy"
  if not os.path.isfile(filename):
    cl = build_crystal(pic, ta_rad=ta_rad, fields=True, height=hslab, onlyfirst=onlyfirst)
    cl.set_source(1.01, te=polars[polar][0],tm=polars[polar][1])
    cl.solve()
    (RT, R), (TT, T) = cl.poynting_flux_end(only_total=False)
    print("total flux:", RT, TT, RT+TT)
    if False:
      fig, axs = plt.subplots(1,2)
      axs[0].matshow(np.abs(R).reshape(5,5), vmax=1, vmin=0)
      axs[1].matshow(np.abs(T).reshape(5,5), vmax=1, vmin=0)
      plt.savefig("debug.png")
      exit()
    cldepth = cl.depth - 4 * dbuffer
    #x, y, z = coords(0.5, 0.5, 0, xcells, dbefore, dbefore+dair+hslab+dafter, (xres, yres, zres))
    #x, y, z = coords(0.5, 0.5, 0, xcells , dbefore, dbefore+dair+hslab+dafter, (xres, yres, zres))
    x = y = np.linspace(0, xcells, xres)
    x, y = np.meshgrid(x, y, indexing="ij")
    z = np.linspace(dbefore, dbefore+dair+hslab+dafter, zres)
    E, H = cl.fields_volume(x,y, z)
    np.save(filename, E)
  else:
    E = np.load(filename)
  print(pic.shape)
  #exit()
  component = 0
  fig, ax = plt.subplots(figsize=(4,3), dpi=300)
  extent = [0, xcells, dbefore+dair+hslab+dafter, dbefore]
  kw = {'cmap': "RdBu", 'origin': "upper", 'aspect': "auto", "extent": extent, "vmin":-1, 'vmax': 1}
  if plane == "x":
    im = ax.matshow(np.real(E[:, component, :, xres//2]), **kw)
  elif plane == "y":
    im = ax.matshow(np.real(E[:, component, xres//2, :]), **kw)
  ax.set_xlim(0, 12)
  c = pic.shape[0]//2
  extent = [0, xcells, dair, dair+hslab/2]
  #ax.contourf((np.tile(pic[:c,:], (1,xcells))), levels=[3,4], extent=extent, colors=['k'], alpha=0.3)
  extent = [0, xcells / np.cos(ta_rad), dair+hslab/2, dair+hslab]
  #ax.contourf((np.tile(pic[c:,:], (1,xcells))), levels=[3,4], extent=extent, colors=['k'], alpha=0.3)
  ax.axhline(dair + hslab, color='k', ls=":")
  ax.axhline(dair, color='k', ls=":")
  ax.axhline(dair+hslab/2, color='k', ls=":")
  
  ax.axis("equal")
  ax.set_xlabel(f"{plane} [µm]")
  ax.set_ylabel("Z [µm]")
  ax.set_xlim(0, xcells)
  plt.colorbar(im)
  filename = f'figs/fields/efield_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}_{round(ta_deg,1)}'
  fig.savefig(f'{filename}.png')
  fig.savefig(f'{filename}.pdf')


def worker(config, twist_angle):
  h, o = config
  xz_pic = parallelogram_inside_slab(o, h, 0.4, eps_parallelogram=4, eps_background=2, hslab=hslab)
  cl = build_crystal(xz_pic, ta_rad=twist_angle, height=hslab)
  cl.set_source(1.01, te=polars[polar][0],tm=polars[polar][1])
  cl.solve()
  (_, _), (_, T) = cl.poynting_flux_end(only_total=False)
  return T.reshape(pw[0], pw[0])

def viz_configs(configs, shape):
  fig, axs = plt.subplots(shape[0], shape[1], dpi=300, figsize=(4,6), gridspec_kw = {'wspace':0, 'hspace':0})
  axs = axs.flatten()
  for (h, o), ax in zip(configs, axs):
    pic = parallelogram_inside_slab(o, h, 0.4, epsilon_parallelogram=4, epsilon_background=2)
    ax.matshow(pic, cmap="Blues", extent=[0, 1, 0, hslab])
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])
  fig.savefig("Parallelograms.png")
  exit()
# Parameter sweep
if program == "sweep":
  id = int(sys.argv[2])
  no, nh = 64, 64
  omin, omax, hmin, hmax = -40, 40, 1.0, hslab
  orientations = np.deg2rad(np.linspace(omin, omax, no))
  heights = np.linspace(hmin, hmax, nh)
  #heights = np.linspace(0.05, 0.95, nh)
  rt = []
  configs = list(product(heights, orientations))
  #viz_configs(configs, (no, nh))
  filename = f"components_study/understand_twisted_spacer.{polar}.{pw[0]}.{id}.npy"
  num_orders = pwt[0] * pwt[1]
  if not os.path.isfile(filename):
    print("File does not exist. Computing.")
    ta_rad = np.deg2rad(59) #np.linspace(0, 60, 10)[id])
    workeri = partial(worker, twist_angle=ta_rad)
    with Pool(4) as p:
      rt = p.map(workeri, configs)
    rt = np.array(rt).reshape(nh, no, num_orders)
    np.save(filename, rt)
  else:
    rt = np.load(filename)
    print(rt.shape)
  mag = np.abs(rt)**2
  phase = np.angle(rt)

  def setlims(ax):
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels([str(omin), "0", str(omax)])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels([np.max(heights),(np.max(heights)+np.min(heights))/2, np.min(heights)])
    ax.xaxis.tick_bottom()
  
  def setup_axes(axs):
    for ax in axs[-1, :]:
      ax.set_xlabel("Slant angle [deg]")
    for ax in axs[:, 0]:
      ax.set_ylabel("Para. Width [um]")
    
    for j, ax in np.ndenumerate(axs):
      ax.set_title(f"{[i-pwt[0]//2 for i in j]}")

  fig, axs = plt.subplots(pwt[0], pwt[1], figsize=(12,12), dpi=300)
  setup_axes(axs)
  axs = axs.flatten()
  for i in range(num_orders):
    im = axs[i].matshow(mag[..., i], extent=[0, 1, 0,1], vmin=0, vmax=1)
    setlims(axs[i])
  fig.subplots_adjust(right=0.9, left=0.08, top=0.95, bottom=0.08)
  cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
  fig.colorbar(im, cax=cbar_ax)
  fig.savefig(f"components_study/OrientationInfluenceSpacer.{polar}.{id}.png")


if False:
  fig, axs = plt.subplots(1, 10, figsize=(8,2))

  for i in range(10):
    d = np.load(f"understand_twisted.te.5.{i}.npy")
    d = d.reshape(64,64,5,5)
    ta = np.linspace(0, 60, 10)[i]
    axs[i].matshow((d[..., 1, 3].real-d[..., 3, 1].real)/(d[..., 1, 3].real+d[..., 3, 1].real), alpha=(d[..., 1, 3].real+d[..., 3, 1].real), vmin=-1, vmax=1, cmap="PuOr") # 8
    #axs[1, i].matshow(d[..., 3, 1].real, vmin=0, vmax=1) # 16
    axs[i].axis("off")
    #axs[1, i].axis("off")
    #axs[i].set_title(f"a={round(ta,3)}")
  plt.tight_layout()
  fig.savefig("TwistingOrders.png")

# Parameter sweep
if program == "fancy":
  id = int(sys.argv[2])
  no, nh = 64, 64
  omin, omax, hmin, hmax = -40, 40, 1.0, hslab
  orientations = np.deg2rad(np.linspace(omin, omax, no))
  heights = np.linspace(hmin, hmax, nh)
  #heights = np.linspace(0.05, 0.95, nh)
  rt = []
  configs = list(product(heights, orientations))
  #viz_configs(configs, (no, nh))
  filename = f"components_study/understand_twisted_spacer.{polar}.{pw[0]}.{id}.npy"
  num_orders = pwt[0] * pwt[1]
  if not os.path.isfile(filename):
    print(f"File {filename} does not exist, error.")
    exit()
  rt = np.load(filename)
  mag = np.abs(rt)**2
  phase = np.angle(rt)

  def setlims(ax):
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels([str(omin), "0", str(omax)])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels([np.max(heights),(np.max(heights)+np.min(heights))/2, np.min(heights)])
    ax.xaxis.tick_bottom()
  
  def setup_axes(axs):
    for ax in axs[:]:
      ax.set_xlabel("Slant angle [deg]")
    for ax in axs[:]:
      #ax.set_ylabel("Para. Width [um]")
      ax.set_ylabel("Para. Height [um]")
    

  fig, axs = plt.subplots(1, pwt[0], figsize=(7,3), dpi=300)
  setup_axes(axs)
  mag = mag.reshape(no, nh, pwt[0], pwt[1])
  axs = axs.flatten()
  c = pwt[0]//2
  im = axs[0].matshow(mag[..., c-1, c+1], extent=[0, 1, 0,1], vmin=0, vmax=1)
  axs[0].set_title("(+1,-1)")
  axs[1].set_title("(0,0)")
  axs[2].set_title("(-1,+1)")
  axs[0].axvline(0.72, color="w", ls="--")
  axs[2].axvline(1-0.72, color="w", ls="--")
  im = axs[1].matshow(mag[..., c, c], extent=[0, 1, 0,1], vmin=0, vmax=1)
  im = axs[2].matshow(mag[..., c+1, c-1], extent=[0, 1, 0,1], vmin=0, vmax=1)
  np.save("twisted.slantheightmappm.npy", mag[..., c-1, c+1])
  for i in range(pwt[0]):
    setlims(axs[i])
  fig.subplots_adjust(right=0.9, left=0.08, top=0.95, bottom=0.08)
  cbar_ax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
  fig.colorbar(im, cax=cbar_ax)
  fig.savefig(f"components_study/OrientationInfluenceSpacer.{polar}.{id}.png")
  fig.savefig(f"components_study/OrientationInfluenceSpacer.{polar}.{id}.pdf")

  fig, ax = plt.subplots()
  x = np.linspace(-45, 45, len(mag[..., c+1, c-1].T[:, 40]))
  ax.plot(x, mag[..., c+1, c-1].T[:, -1])
  print(len(x))
  xr = np.deg2rad(-22-x)
  print(np.rad2deg(0.15))
  ax.plot(x, np.exp(-xr**2/0.08**2)*0.6)
  fig.savefig("Efficiency.png")