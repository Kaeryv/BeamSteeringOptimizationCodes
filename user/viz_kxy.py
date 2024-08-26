from bast.expansion import Expansion
import matplotlib.pyplot as plt
import numpy as np
pw = (5,1)
e1 = Expansion(pw)
e2 = Expansion(pw)
e2.rotate(30)
e = e1 + e2
gx, gy = e.g_vectors
selected_order = 8

fig, (ax,ax2) = plt.subplots(1,2, figsize=(8,5))
tpi = 2*np.pi
ax.scatter(gx.flatten()/tpi, gy.flatten()/tpi, color="red", facecolors="none")
#for i, (x,y) in enumerate(zip(gx.flatten(), gy.flatten())):
#    ax.text(x,y,f"{i}")
ax.scatter(gx.flatten()[selected_order]/tpi, gy.flatten()[selected_order]/tpi, color='b')
ax.axis("equal")
fig.savefig("kxy.png")

#fig, ax = plt.subplots()
kxytraj = []
k = []
thetas = np.linspace(0,60)
for theta in thetas:
    e1 = Expansion(pw)
    e2 = Expansion(pw)
    e2.rotate(theta)
    e = e1 + e2
    gx, gy = e.g_vectors
    k.append([kk.flatten()[selected_order] for kk  in e.k_vectors((0,0), 1.0)])
    kxytraj.append((gx.flatten()[selected_order]/tpi, gy.flatten()[selected_order]/tpi))
k = np.array(k)
kudot = k.dot(np.array([0,0,1]))
kxytraj = np.array(kxytraj)
ax.plot(*kxytraj.T, 'b-')
polar = np.rad2deg(np.arccos(kudot.real))
ax2.plot(thetas, polar)
circ_x = np.cos(np.linspace(0, 2*np.pi))
circ_y = np.sin(np.linspace(0, 2*np.pi))
ax.plot(circ_x,circ_y, 'k:')
ax.axis("equal")
ax2.axis("equal")
ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax2.set_xlabel("Twist angle")
ax2.set_ylabel("Polar angle")
ax2.axvline(thetas[np.argmax(polar)])
ax2.text(thetas[np.argmax(polar)], 60, round(thetas[np.argmax(polar)],2))
ax.set_xticks([-4,-2, 0, 2, 4])
ax.set_yticks([-2, 0, 2])
ax.set_xticklabels([f"{2*i} $\pi$" for i in [-4,-2, 0, 2, 4]])
ax.set_yticklabels([f"{2*i} $\pi$" for i in [-2, 0, 2]])
plt.tight_layout()
fig.savefig("kxy_traj.png")
fig.savefig("kxy_traj.pdf")