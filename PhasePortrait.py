import numpy as np
import matplotlib.pyplot as plt

def strogatz(x,y):
    a = 1
    b = 1
    return [a - x - 4 * x * y / (1+ x*x), b * x * (1 - y / (1 + x*x))]

x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)

ver = np.zeros(Y.shape)
hor = np.zeros(Y.shape)
NI, NJ = ver.shape
for i in range(NI):
    for j in range(NJ):
        x = X[i, j]
        y = Y[i, j]
        uv = strogatz(x,y)
        hor[i,j] = uv[0]
        ver[i,j] = uv[1]

    
Q = plt.quiver(X, Y, hor, ver, color='r')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig('phase-portrait.png')