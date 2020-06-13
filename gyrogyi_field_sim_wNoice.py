import numpy as np
import matplotlib.pyplot as plt
import math
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

noice_level = 2


dt = 0.5
n_itr = math.floor(T/dt)

sol_x = [None]*n_itr
sol_z = [None]*n_itr
sol_v = [None]*n_itr


for i in range (0,n_itr-1):
    
    flow_dif = np.random.normal(loc=0, scale=noice_level)
    gf = GyorgyiFieldModel(kf+flow_dif)
    
    sol = solve_ivp(
    gf.rhs, (0, dt), np.array([x0, z0, v0]), method="DOP853", atol=tol, rtol=tol
    )

    x0 = sol.y[0,-1]
    z0 = sol.y[1,-1]
    v0 = sol.y[2,-1]

    sol_x[i] = sol.y[0,:]
    sol_z[i] = sol.y[1,:]
    sol_v[i] = sol.y[2,:]


x = []
z = []
v = []

for k in range (0,n_itr-1):
    np.append(x, sol_x[k])
    np.append(z, sol_z[k])
    np.append(v, sol_v[k])

