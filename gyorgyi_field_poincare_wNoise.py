import numpy as np
import matplotlib.pyplot as plt
import math
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
surfx = 0.0468627

gf = GyorgyiFieldModel(kf)


# Poincaré section when x = surfx
def poincare_surf(t, Y):
    return Y[0] - surfx


# Only points with decreasing x
poincare_surf.direction = -1
# Skip first few transient events
sol = solve_ivp(
    gf.rhs,
    (0, 0.1),
    np.array([x0, z0, v0]),
    method="DOP853",
    events=poincare_surf,
    atol=tol,
    rtol=tol,
)

x0 = sol.y[0, -1]
z0 = sol.y[1, -1]
v0 = sol.y[2, -1]

dt = 0.002
nbr_itr = math.floor((T - 0.1) / dt)
noise_level = 0.1
t = 0.1
poincare = np.vstack(sol.y_events[0][sol.t_events[0] >= 0.08]).T
poincare = np.append(poincare, poincare, 1)
for i in range(0, nbr_itr):
    flow_diff = np.random.normal(loc=0, scale=noise_level)
    gf = GyorgyiFieldModel(kf + kf * flow_diff)

    sol = solve_ivp(
        gf.rhs,
        (t, t + dt),
        np.array([x0, z0, v0]),
        method="DOP853",
        events=poincare_surf,
        atol=tol,
        rtol=tol,
    )
    t = t + dt
    x0 = sol.y[0, -1]
    z0 = sol.y[1, -1]
    v0 = sol.y[2, -1]
    if sol.y_events[0].size != 0:
        poincaretemp = np.vstack(sol.y_events[0][sol.t_events[0] >= 0.1]).T
        poincare = np.append(poincare, poincaretemp, 1)

plt.scatter(poincare[1, :], poincare[2, :])
plt.xlabel("$z$")
plt.ylabel("$v$")
plt.show()

return_from = poincare[1, :-1]
return_to = poincare[1, 1:]

plt.scatter(return_from, return_to)
plt.xlabel("$z_i$")
plt.ylabel("$z_{i+1}$")
plt.show()
