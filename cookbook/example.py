
from ljhouses import _simulation, StochasticBerendsenThermostat, NVEThermostat
from ljhouses.tools import get_lattice_initial_conditions

import numpy as np

N = 200
LJ_r = 6
LJ_e = 2
LJ_Rmax = 4*6
g = 0.1
v0 = 0
dt = 0.01

x, v, a = get_lattice_initial_conditions(N, v0, LJ_r)
thermostat = NVEThermostat()
thermostat = StochasticBerendsenThermostat(10.0, N)

samples, t, K, V, Vij = _simulation(dt, 1_000, 1, 10, LJ_r, LJ_e, LJ_Rmax, g, x, v, a, thermostat)
print(samples[-1][0][:2])
print(samples[-2][0][:2])


import matplotlib.pyplot as pl

pl.figure()
pl.plot(t, K)
pl.plot(t, V)
pl.plot(t, Vij)
pl.plot(t, np.sum((K, V, Vij),axis=0))

pl.show()
