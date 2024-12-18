from bast.expansion import Expansion
import matplotlib.pyplot as plt
import numpy as np

pw = (5,1)

def get_e(alpha):
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    alpha_disp = np.deg2rad(alpha)
    e1.rotate(-alpha_disp/2)
    e2.rotate(+alpha_disp/2)
    e = e1 + e2
    gx, gy = e.g_vectors
    return gx, gy, e, e1, e2
selected_order = 18

fig, (ax,ax2) = plt.subplots(1,2, figsize=(5,2.5))
tpi = 2*np.pi
#for i, (x,y) in enumerate(zip(gx.flatten(), gy.flatten())):
#    ax.text(x,y,f"{i}")
def unitcircle(ax):
    circ_x = np.cos(np.linspace(0, 2*np.pi))
    circ_y = np.sin(np.linspace(0, 2*np.pi))
    ax.plot(circ_x,circ_y, linestyle=':', zorder=-1, color="gray")

kxytraj = []
k = []
N = 200
thetas = np.deg2rad(np.linspace(0,60, N))
for theta in thetas:
    gx, gy, e, _, _ = get_e(np.rad2deg(theta))
    k.append([kk.flatten()[selected_order] for kk  in e.k_vectors((0,0), 1.0)])
    kxytraj.append([g.flatten()[selected_order]/tpi for g in e.g_vectors])
k = np.array(k)
print(k.shape)
kudot = k.dot(np.array([0,0,1]))
print(kudot.shape)
kxytraj = np.array(kxytraj)
unitcircle(ax)

gx, gy, e, e1, e2 = get_e(alpha=40)
ax.plot(*kxytraj.T, 'b-',zorder=0)
mask = (gx**2+gy**2 <= 2**2*tpi**2)
ax.scatter(gx[mask].flatten()/tpi, gy[mask].flatten()/tpi, color="red", facecolors="w", marker="o", s=80)
ax.scatter(gx.flatten()[selected_order]/tpi, gy.flatten()[selected_order]/tpi, facecolors="none", marker="o", s=100, color='b')
gx, gy = e1.g_vectors
mask = (gx**2+gy**2 <= 2**2*tpi**2)
ax.scatter(gx[mask].flatten()/tpi, gy[mask].flatten()/tpi, marker="+", color="k", facecolors="k")
gx, gy = e2.g_vectors
mask = (gx**2+gy**2 <= 2**2*tpi**2)
ax.scatter(gx[mask].flatten()/tpi, gy[mask].flatten()/tpi, marker="x", color="k", facecolors="k")
ax.axis("equal")
polar = np.rad2deg(np.arccos(kudot.real))


ax.axis("equal")
ax.set_xlabel("$k_x [\\frac{2\\pi}{a}]$"), ax.set_ylabel("$k_y[\\frac{2\\pi}{a}]$")
xticks=yticks=[-2, 0, 2]
ax.set_xticks(xticks), ax.set_yticks(yticks)

ax2.plot(np.rad2deg(thetas), polar)
#ax2.axis("equal")
ax2.set_xlabel("Twist angle"), ax2.set_ylabel("Polar angle")
ax2.set_xticks(np.arange(0,61, 20))
ax2.set_yticks(np.arange(0,91, 15))
ax2.set_xlim(-1, 61)
ax2.set_ylim(-1, 95)
plt.tight_layout()
fig.savefig("kxy_traj.png")
fig.savefig("kxy_traj.pdf")