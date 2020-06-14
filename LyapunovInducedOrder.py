import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Lyapunov_rescaled import est_lyapunov

kf = 3.4e-4
num_points = 10
noise = np.linspace(0, 0.1, num_points)

# print(noise)

lyapunov = np.zeros(num_points)


for i in range(0, num_points):
    lya = 0
    lya += est_lyapunov(kf, noise[i])
    lya += est_lyapunov(kf, noise[i])
    lya += est_lyapunov(kf, noise[i])
    lya += est_lyapunov(kf, noise[i])
    lyapunov[i] = lya / 4

plt.plot(noise, lyapunov)
plt.xlabel("Noise level")
plt.ylabel("Estimated largest Lyapunov exp.")
plt.show()
