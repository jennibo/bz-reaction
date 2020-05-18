import numpy as np
import matplotlib.pyplot as plt
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

sol = solve_ivp(
    gf.rhs,
    (0, T),
    np.array([x0, z0, v0]),
    method="DOP853",
    events=poincare_surf,
    atol=tol,
    rtol=tol,
)

# Skip first few transient events
skip_until_t = 0.1
poincare = np.vstack(sol.y_events[0][sol.t_events[0] >= skip_until_t]).T

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
