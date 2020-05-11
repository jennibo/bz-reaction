import numpy as np

k1 = 4e6
k2 = 2.0
k3 = 3000.0
k4 = 55.2
k5 = 7000.0
k6 = 0.09
k7 = 0.23

A = 0.1
M = 0.25
H = 0.26
C = 0.000833
alpha = 666.7
beta = 0.3478

T0 = 1 / (10 * k2 * A * H * C)
X0 = k2 * A * H * H / k5
Y0 = 4 * X0
Z0 = C * A / 40
V0 = 4 * A * H * C / (M * M)


class GyorgyiFieldModel:
    def __init__(self, kf):
        self.kf = kf

    def __y_tilde(self, x, z, v):
        return (
            alpha * k6 * Z0 * V0 * z * v / (k1 * H * X0 * x + k2 * A * H * H + self.kf)
        ) / Y0

    def rhs(self, t, Y):
        x, z, v = Y
        y_tilde = self.__y_tilde(x, z, v)
        kf = self.kf
        return T0 * np.array(
            [
                -k1 * H * Y0 * x * y_tilde
                + k2 * A * H * H * Y0 / X0 * y_tilde
                - 2 * k3 * X0 * x * x
                + 0.5 * k4 * (C - Z0 * z) * H * np.sqrt(np.abs(A * H * x / X0))
                - 0.5 * k5 * Z0 * x * z
                - kf * x,
                k4 * (C / Z0 - z) * H * np.sqrt(np.abs(A * H * X0 * x))
                - k5 * X0 * x * z
                - alpha * k6 * V0 * z * v
                - beta * k7 * M * z
                - kf * z,
                2 * k1 * H * X0 * Y0 / V0 * x * y_tilde
                + k2 * A * H * H * Y0 / V0 * y_tilde
                + k3 * X0 * X0 / V0 * x * x
                - alpha * k6 * Z0 * z * v
                - kf * v,
            ]
        )
