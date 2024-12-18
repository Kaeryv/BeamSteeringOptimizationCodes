import numpy as np
from user.understand.model import get_polar

L = 1.0
m1 = -1
m2 = 1
geeff = 2
N = 100
alphas = np.linspace(0.001, np.pi/3, N)
def get1D(nt2, nt=np.sqrt(geeff), l=1.01):
    tt1 = np.arcsin(-m1 * l / L / nt)
    sol = list()
    tt2 = np.arcsin(
            l/L*np.sqrt(
            + (-m1 - m2*np.cos(alphas))**2 
            #+ (np.sin(alphas))**2
            )
        )
    sol = np.rad2deg(tt2)
    return sol
def get(nt2, nt=np.sqrt(geeff), l=1.01):
    tt1 = np.arcsin(-m1 * l / L / nt)
    #nt2 = np.sqrt(1)
    sol = list()
    
    #tt2 = np.arcsin(
    #        np.sqrt(
    #        + (nt * np.sin(tt1) - m2*l/L*np.cos(alphas))**2 
    #        + (l/L*np.sin(alphas))**2
    #        )
    #    )
    tt2 = np.arcsin(
            l/L*np.sqrt(
            + (-m1 - m2*np.cos(alphas))**2 
            + (np.sin(alphas))**2
            )
        )
    #tt2 = np.arcsin(
    #        l/L*
    #       (-m1 - m2*np.cos(alphas))
    ##        + (np.sin(alphas))**2
    #    )
    #tt2 = np.arcsin(
    #        l/L*np.sqrt(
    #            2 * (1 - np.cos(alphas))
    #        ) 
    #    )
    return tt2



gdisp =22.5
pout,phis = get_polar(gdisp)
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(3.2,6))
ax1.plot(np.rad2deg(alphas), np.rad2deg(get(1, l=1.01)), label="Diffraction")
ax1.plot(np.rad2deg(phis), np.rad2deg(pout), 'r', label="Blazing")
ax1.set_xlabel("Twist angle [deg]")
ax1.set_ylabel("Polar angle [deg]")
from user.commons import load_variable
d = load_variable("sim_data.pkl")
x = np.linspace(0,60, len(d.metric[0]))
sigma=0.15

I = 0.9**2*np.exp(-(get(1,l=1.01) - pout)**2/(2*sigma**2))
ax1b=ax1.twinx()
ax1b.plot(np.rad2deg(alphas), I, label="Model", color="k", ls="-")
ax1b.plot(x, d.metric.T[::1,0], label="RCWA", ls=":", color="k")
ax1b.set_ylim(0,1.1)
ax1.legend()

gammas = np.linspace(-45, 45, 100)
mean_efficiency_vs_gamma = np.zeros_like(gammas)
diffraction_angle = get(1,l=1.01)
efficiency_vs_gamma_twist = np.zeros((len(gammas), len(diffraction_angle)))
for i, g0 in enumerate(gammas):
    sigma=0.3

    efficiency1 = 0.83*np.exp(-(np.deg2rad(2*g0-45))**2/sigma**2)

    blaze, phis = get_polar(g0)
    #blaze[np.isnan(blaze)] = np.deg2rad(2*g0)
    efficiency2 = 0.83*np.exp(-(diffraction_angle - blaze)**2/(2*sigma**2))
    efficiency2[np.isnan(blaze) | np.isnan(efficiency2)] = 0
    efficiency = efficiency1* efficiency2
    mean_efficiency = np.mean(efficiency)
    mean_efficiency_vs_gamma[i] = mean_efficiency
    efficiency_vs_gamma_twist[i, :] = efficiency.copy()

plt.legend()
"""
    Efficiency function of the slant angle of the parallelogram.
    We compare RCWA results to the analytical model.
"""
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def custom_red_to_green_colormap():
    """
    Creates a colormap that transitions from slightly dark red to slightly dark green.

    Returns:
        LinearSegmentedColormap: A matplotlib colormap object.
    """
    colors = [
        '#5E81AC',  # Slightly dark red (RGB)
        #'#D08770',
        '#BF616A'  # Slightly dark green (RGB)
    ]

    cmap = LinearSegmentedColormap.from_list("RedToGreen", colors, N=256)
    return cmap
rcwa2 = [np.load(f"components_study/understand_twisted_spacer.te.3.{160+i}.npy") for i in range(11)]
rcwa2 = np.array(rcwa2).reshape(-1, 64,64,3,3)[..., 0, 2]
rcwa2mean = np.mean(rcwa2, axis=0)
ntwist = efficiency_vs_gamma_twist.shape[-1]

colors = custom_red_to_green_colormap()(np.linspace(0, 1, len(rcwa2)))
colors2 = custom_red_to_green_colormap()(np.linspace(0, 1, ntwist))
#for i in range(0, ntwist, 10):
#    ax2.plot(gammas, efficiency_vs_gamma_twist[:, i], color=colors2[i])
for i in range(0, len(rcwa2), 2):
    ax2.plot(np.linspace(-40, 40, len(rcwa2mean)),rcwa2[i, -1], color=colors[i])
ax2.plot(gammas, mean_efficiency_vs_gamma, 'k', label="Reduced model")

print(gammas[np.nanargmax(mean_efficiency_vs_gamma)])
#mx = gammas[np.nanargmax(mean_efficiency_vs_gamma)]
#ax2.axvline(mx)
ax2.set_xlabel("Slant angle [deg]")
ax2.set_ylabel("Twist-averaged diffraction efficiency")
ax2.legend()
plt.savefig("figs/BlazingModel.png")
plt.savefig("figs/BlazingModel.pdf")