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
    # T = 1
    T = 10
    x0 = 0.05
    z0 = 9.0
    v0 = 0.85
    tol = 1e-8

    diff = 1e-10
    dt = 0.002
    nbr_itr = math.floor(T / dt)

    lyapunov = 0

    u = np.array([x0, z0, v0])
    v = np.array([x0 + diff, z0, v0])

    gf = GyorgyiFieldModel(kf)

    solver_u = DOP853(gf.rhs, 0, u, dt, atol=tol, rtol=tol)
    solver_v = DOP853(gf.rhs, 0, v, dt, atol=tol, rtol=tol)
    for i in range(0, nbr_itr):

        flow_diff = np.random.normal(loc=0, scale=noise_level)
        gf.kf = kf + kf * flow_diff

        solver_u.y = u
        solver_u.t_bound = (i + 1) * dt
        solver_u.status = "running"
        while solver_u.status == "running":
            solver_u.step()
        u = solver_u.y

        solver_v.y = v
        solver_v.t_bound = (i + 1) * dt
        solver_v.status = "running"
        while solver_v.status == "running":
            solver_v.step()
        v = solver_v.y

        delta = v - u
        d = np.linalg.norm(delta)
        if i >= 200:
            lyapunov += np.log(d / diff)

        v = u + diff * delta / d

    lyapunov = lyapunov / (dt * (nbr_itr - 200))
    return lyapunov


if __name__ == "__main__":
    print(est_lyapunov(kf, noise_level))
