import numpy as np
import matplotlib.pyplot as plt

angles = list()
for i in range(81):
    File = f"data/ellipsisd32_1_16_pso/free_pixmap_{i}/best.npz"

    data = np.load(File)
    angles.append(data["bd"][-1])

angles = np.arctan(3.2* np.tan(np.abs(angles)))
plt.hist(90 - np.rad2deg(angles))
plt.savefig("test.png")

