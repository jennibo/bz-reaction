import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from gyorgyi_field import GyorgyiFieldModel

kf = 3.23e-4
T = 0.5
x0 = 0.05
z0 = 9.0
v0 = 0.85
tol = 1e-8

gf = GyorgyiFieldModel(kf)
Y0 = np.array([x0, z0, v0])
l = 5  # Meshgrid = l*l

sol = solve_ivp(
    gf.rhs, (0, T), np.array([x0, z0, v0]), method="DOP853", atol=tol, rtol=tol
)
x = np.linspace(0.01, 0.56, l)
z = np.linspace(np.power(10, 0.3), np.power(10, 1), l)
v = np.linspace(0.845, 0.852, l)

colors = ["r", "g", "b", "c", "m", "y", "k"]

# Calculates and plots the X-Z phase portrait
X, Y = np.meshgrid(x, z)  # "X and Y axis" in the phase portrait
x_values = np.zeros(Y.shape)
z_values = np.zeros(Y.shape)
fig = plt.figure()
for h in range(l):
    vtemp = v[h]
    for i in range(l):
        for j in range(l):
            xtemp = X[i, j]
            ztemp = Y[i, j]
            xzv = np.array([xtemp, ztemp, vtemp])
            uv = gf.rhs(0, xzv)
            x_values[i, j] = uv[0]
            z_values[i, j] = uv[1]
    plt.quiver(X, Y, x_values, z_values, color=colors[h])
plt.plot(sol.y[0, :], sol.y[1, :])
plt.show()

# Calculates and plots the X-V phase portrait
X, Y = np.meshgrid(x, v)
x_values = np.zeros(Y.shape)
v_values = np.zeros(Y.shape)
fig = plt.figure()

for h in range(l):
    ztemp = z[h]
    for i in range(l):
        for j in range(l):
            xtemp = X[i, j]
            vtemp = Y[i, j]
            xzv = np.array([xtemp, ztemp, vtemp])
            uv = gf.rhs(0, xzv)
            x_values[i, j] = uv[0]
            v_values[i, j] = uv[2]
    plt.quiver(X, Y, x_values, v_values, color=colors[h])
plt.plot(sol.y[0, :], sol.y[2, :])
plt.show()

# Calculates and plots the Z-V phase portrait
X, Y = np.meshgrid(z, v)
z_values = np.zeros(Y.shape)
v_values = np.zeros(Y.shape)
fig = plt.figure()
for h in range(l):
    xtemp = x[h]
    for i in range(l):
        for j in range(l):
            ztemp = X[i, j]
            vtemp = Y[i, j]
            xzv = np.array([xtemp, ztemp, vtemp])
            uv = gf.rhs(0, xzv)
            z_values[i, j] = uv[1]
            v_values[i, j] = uv[2]
    plt.quiver(X, Y, z_values, v_values, color=colors[h])
plt.plot(np.abs(sol.y[1, :]), np.abs(sol.y[2, :]))
plt.show()
