import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from gyorgyi_field import GyorgyiFieldModel
from numpy import random


flow_dif = random.normal(size=(2, 3))
flow_crit = 3.4e-4

T = 2
x0 = 0.05
z0 = 9.0
v0 = 0.85
tol = 1e-8

diff = 1e-9

xprime_0 = x0+ diff
zprime_0 = z0 + diff
vprime_0 = v0 + diff






