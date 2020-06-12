import numpy as np
import math
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

diff = 1e-12

gf = GyorgyiFieldModel(kf)

sol = solve_ivp(
    gf.rhs, (0, T), np.array([x0, z0, v0]), method="DOP853", atol=tol, rtol=tol
)

sol1_x = sol.y[0,:]
sol1_z = sol.y[1,:]
sol1_w = sol.y[2,:]

sol = solve_ivp(
    gf.rhs, (0, T), np.array([x0+diff, z0, v0]), method="DOP853", atol=tol, rtol=tol
)

sol2_x = sol.y[0,:]
sol2_z = sol.y[1,:]
sol2_w = sol.y[2,:]

delta = [0]*len(sol2_x)
#detla_y = [None]*len(sol1_x)
#delta_z = [None]*len(sol1_x)


for i in range (0,len(sol2_x)-1):
    x = sol1_x[i]-sol2_x[i]
    w = sol1_z[i]-sol2_z[i]
    z =sol1_w[i]-sol2_w[i]
    delta[i] = math.sqrt(x**2+w**2+z**2)

fig = plt.figure()
plt.plot(sol.t, np.log(delta))
plt.xlabel("t")
plt.ylabel("lg(-)")
plt.show()
