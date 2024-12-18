import matplotlib.pyplot as plt
import numpy as np

def plotvec(ax, v, c='k', x0=[0,0,0], ls=None):
    ax.plot([x0[0],x0[0]+v[0]],[x0[1],x0[1]+v[1]],[x0[2],x0[2]+v[2]], c=c, ls=ls)
def cyl2xyz(v):
    return np.array([
        v[0]*np.cos(v[1]),
        v[0]*np.sin(v[1]),
        v[2]])
def plotveccyl(ax, v, *kwargs):
    plotvec(ax, cyl2xyz(v), *kwargs)

def plotplane(ax, x, normal):
    n_1, n_2, n_3 = normal
    x_0, y_0, z_0 = x
    x = np.arange(-0.5, 0.5, 0.1)
    y = np.arange(-0.5, 0.5, 0.1)
    xg, yg = np.meshgrid(x, y)

    # Compute z = f(x,y)

    z = -n_1/n_3*(xg-x_0)-n_2/n_3*(yg-y_0)+z_0 

    ax.plot_wireframe(xg, yg, z, rstride=10, cstride=10)

def reflect(v, n):
    return v - np.dot(n, v) * 2 * n

# def refract(v, n, n1, n2):
#     nv = np.dot(n, v)
#     return n1/n2*((np.sqrt(nv**2+(n2/n1)**2-1)-nv) * n + v)

def refract(v, n, n1, n2):
    nv = - np.dot(n, v)
    sin2 = (n1/n2)**2*(1-nv**2)
    return n1/n2 * v + (n1/n2 *nv - np.sqrt(1-sin2)) * n


def get_polar(gamma0, incident=45, samples=100):
  gamma = np.deg2rad(90+gamma0)
  ti = np.deg2rad(incident)
  z0 = np.cos(gamma)
  r = np.sin(gamma)
  phis = np.linspace(0, np.pi/3, samples)
  pout = list()
  for phi in phis:
    n = cyl2xyz([r, phi, z0])
    n /= np.linalg.norm(n)
    d = -np.array([np.sin(ti), 0, np.cos(ti)])
    d /= np.linalg.norm(d)
    refl = reflect(d, n)
    polar_out = np.arccos(np.dot(refl, [0,0,-1]) / np.linalg.norm(refl))
    pout.append(polar_out)

  pout = np.arcsin(np.sin(pout) * np.sqrt(2))
  return pout, phis

if __name__ == "__main__":
    ax = plt.figure().add_subplot(projection='3d')
    #ax.set_proj_type('ortho')
    neff = np.sqrt(2)
    gamma = np.deg2rad(90+22.5)
    ti = np.deg2rad(45)
    print("ti:", np.rad2deg(ti))
    z0 = np.cos(gamma)
    r = np.sin(gamma)
    phis = np.linspace(0, np.pi/3, 35)
    cols = plt.cm.jet(phis/np.max(phis))    
    plt.axis("equal")
    pout = list()
    for phi, col in zip(phis, cols):
        n = cyl2xyz([r, phi, z0])
        n /= np.linalg.norm(n)
        d = -np.array([np.sin(ti), 0, np.cos(ti)])
        d /= np.linalg.norm(d)
        plotvec(ax, n, "k")
        refl = reflect(d, n)
        #refr = refract(d, n, np.sqrt(neff), 1)
        plotvec(ax, refl, c=col, ls=":")
        #plotvec(ax, refl, c=col)
        plotvec(ax, d, x0=-d)
        polar_out = np.arccos(np.dot(refl, [0,0,-1]) / np.linalg.norm(refl))
        pout.append(polar_out)
        #plotplane(ax, [0,0,0], n)
    ax.legend()
    ax.set_aspect("equal")
    pout = np.arcsin(np.sin(pout) * np.sqrt(2))
    plt.show()
    
    plt.figure()
    #gamma0 = np.deg2rad(90-np.rad2deg(gamma))
    agamma = np.arctan(np.tan(gamma)*np.cos(phis))
    plt.plot(np.rad2deg(phis), np.rad2deg(pout), label=" POUT")
    np.savez_compressed("2Dmodel.npz", phis=phis, pout=pout)
    #plt.plot(np.rad2deg(phis), np.rad2deg(agamma))
    #apout = agamma - np.arcsin(np.sin(agamma-ti)*np.sqrt(2))
    #plt.plot(np.rad2deg(phis), np.rad2deg(apout), label="Analytical POUT", ls=":")
    plt.legend()
    #plt.axhline(theta)
    plt.show()
