import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from gyorgyi_field import GyorgyiFieldModel

#kf = 2.0e-4  # fixed point
# kf = 3.1e-4  # 1-periodic
# kf = 3.2e-4  # 2-periodic
# kf = 3.23e-4  # 4-periodic
# kf = 3.26e-4  # chaotic
# kf = 3.3e-4  # 3-periodic
#kf = 3.4e-4  # chaotic

kf = 100

T = 2
x0 = 0.05
z0 = 9.0
v0 = 0.85
tol = 1e-8

diff = 1e-10
dt = 0.01
nbr_itr = math.floor(T/dt)

delta = [0]*nbr_itr

for i in range (1,nbr_itr):

    gf = GyorgyiFieldModel(kf)

    sol = solve_ivp(
        gf.rhs, (0, dt), np.array([x0, z0, v0]), method="DOP853", atol=tol, rtol=tol
    )

    sol1_x = sol.y[0,:]
    sol1_z = sol.y[1,:]
    sol1_w = sol.y[2,:]

    x1 = sol1_x[-1]
    z1 = sol1_z[-1]
    w1 = sol1_w[-1]

    sol = solve_ivp(
        gf.rhs, (0, dt), np.array([x0, z0+diff, v0]), method="DOP853", atol=tol, rtol=tol
    )

    sol2_x = sol.y[0,:]
    sol2_z = sol.y[1,:]
    sol2_w = sol.y[2,:]

    x2 = sol2_x[-1]
    z2 = sol2_z[-1]
    w2 = sol2_w[-1]
    
    delta[i] = math.sqrt((x1-x2)**2+(w1-w2)**2+(z1-z2)**2)

    x0 = x1
    z0 = z1
    w0 = w1


sum = 0
for q in range(1,len(delta)-1):
    val = delta[q]/diff
    sum += np.log(val)

#Gets delta[0] = 0, Did I mess up above?
print(sum/T)
