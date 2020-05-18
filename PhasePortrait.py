import numpy as np
import matplotlib.pyplot as plt

# Returns the equations from the book by Strogatz
def strogatz(Y):
    a = 10
    b = 2
    x, y = Y
    return np.array([a - x - 4 * x * y / (1 + x * x), b * x * (1 - y / (1 + x * x))])


# Creates a meshgrid
xaxis = 5
yaxis = 10

x = np.linspace(0, xaxis, 20)
y = np.linspace(0, yaxis, 20)
X, Y = np.meshgrid(x, y)
ver = np.zeros(Y.shape)
hor = np.zeros(Y.shape)
NI, NJ = ver.shape
# Calculates the gradients in every point of the meshgrid
for i in range(NI):
    for j in range(NJ):
        x = X[i, j]
        y = Y[i, j]
        uv = strogatz([x, y])
        hor[i, j] = uv[0]
        ver[i, j] = uv[1]

# Plots the phase portrait
Q = plt.quiver(X, Y, hor, ver, color="r")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.xlim([0, xaxis])
plt.ylim([0, yaxis])

# initial values [x0,y0] and t0
y0 = np.array([0, 0])
# t = 0
# Runge-Kutta method to solve paths
h = 0.01  # step size
steps = 10000
Y = np.zeros((steps, 2))
Y[1, :] = y0
for i in range(steps - 1):
    # t = t + h
    k1 = strogatz(Y[i, :])
    k2 = strogatz(Y[i, :] + h / 2 * k1)
    k3 = strogatz(Y[i, :] + h / 2 * k2)
    k4 = strogatz(Y[i, :] + h * k3)
    yn = Y[i, :] + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    Y[i + 1, :] = yn

plt.plot(Y[:, 0], Y[:, 1], "b-")  # path
plt.plot([Y[0, 0]], [Y[0, 1]], "o")  # start
plt.plot([Y[-1, 0]], [Y[-1, 1]], "s")  # end

plt.savefig("phase-portrait.png")
