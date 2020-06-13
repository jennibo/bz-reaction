import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import DOP853
from gyorgyi_field import GyorgyiFieldModel

# kf = 2.0e-4  # fixed point
# kf = 3.1e-4  # 1-periodic
# kf = 3.2e-4  # 2-periodic
# kf = 3.23e-4  # 4-periodic
# kf = 3.26e-4  # chaotic
# kf = 3.3e-4  # 3-periodic
kf = 3.4e-4  # chaotic

noise_level = 0.001


def est_lyapunov(kf, noise_level):
    T = 10
    x0 = 0.05
    z0 = 9.0
    v0 = 0.85
    tol = 1e-8

    diff = 1e-10
    dt = 0.002
    nbr_itr = math.floor(T / dt)

    lyaponov = 0

    u = np.array([x0, z0, v0])
    v = np.array([x0 + diff, z0, v0])

    for i in range(0, nbr_itr):

        flow_diff = np.random.normal(loc=0, scale=noise_level)
        gf = GyorgyiFieldModel(kf + kf * flow_diff)

        solver = DOP853(gf.rhs, 0, u, dt, atol=tol, rtol=tol)
        while solver.status == "running":
            solver.step()
        u = solver.y

        solver = DOP853(gf.rhs, 0, v, dt, atol=tol, rtol=tol)
        while solver.status == "running":
            solver.step()
        v = solver.y

        delta = v - u
        d = np.linalg.norm(delta)
        if i >= 200:
            lyaponov += np.log(d / diff)

        v = u + diff * delta / d

    lyaponov = lyaponov / (dt * (nbr_itr - 200))
    return lyaponov


if __name__ == "__main__":
    print(est_lyapunov(kf, noise_level))
