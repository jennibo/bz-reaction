import numpy as np
import matplotlib.pyplot as plt
import math
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

T = 2
x0 = 0.05
z0 = 9.0
v0 = 0.85
tol = 1e-8

noise_level = 0.1


dt = 0.0002
n_itr = math.floor(T / dt)

sol_x = [0] * n_itr
sol_z = [0] * n_itr
sol_v = [0] * n_itr


for i in range(0, n_itr):

    flow_diff = np.random.normal(loc=0, scale=noise_level)
    gf = GyorgyiFieldModel(kf + kf * flow_diff)

    solver = DOP853(gf.rhs, 0, np.array([x0, z0, v0]), dt, atol=tol, rtol=tol)
    while solver.status == "running":
        solver.step()

    x0 = solver.y[0]
    z0 = solver.y[1]
    v0 = solver.y[2]

    sol_x[i] = solver.y[0]
    sol_z[i] = solver.y[1]
    sol_v[i] = solver.y[2]

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot(
    np.log10(np.abs(sol_x)), np.log10(np.abs(sol_z)), np.log10(np.abs(sol_v)),
)
ax.set_xlabel("lg(x)")
ax.set_ylabel("lg(z)")
ax.set_zlabel("lg(v)")
plt.show()
