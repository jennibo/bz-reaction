import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
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

gf = GyorgyiFieldModel(kf)

sol = solve_ivp(
    gf.rhs, (0, T), np.array([x0, z0, v0]), method="DOP853", atol=tol, rtol=tol
)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot(
    np.log10(np.abs(sol.y[0, :])),
    np.log10(np.abs(sol.y[1, :])),
    np.log10(np.abs(sol.y[2, :])),
)
ax.set_xlabel("lg(x)")
ax.set_ylabel("lg(z)")
ax.set_zlabel("lg(v)")
plt.show()

fig = plt.figure()
plt.plot(sol.t, np.log10(np.abs(sol.y[0, :])))
plt.plot(sol.t, np.log10(np.abs(sol.y[1, :])))
plt.plot(sol.t, np.log10(np.abs(sol.y[2, :])))
plt.xlabel("t")
plt.ylabel("lg(-)")
plt.show()
