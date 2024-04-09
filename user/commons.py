import numpy as np

default_harmonics = np.arange(1, 6)
def freq2pix(amplitudes, phases, resolution=128, harmonics=default_harmonics, threshold=0.2):
    assert len(amplitudes) == len(harmonics)
    assert len(phases) == len(harmonics)
    theta = np.linspace(-np.pi, np.pi, resolution)
    signal = np.zeros_like(theta)
    for f, a, p in zip(harmonics, amplitudes, phases):
        signal += a * np.sin(f * theta + p) 

    return theta, (signal**2 > threshold).astype(np.float64)
