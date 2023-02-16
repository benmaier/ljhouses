
from ljhouses import _simulation, StochasticBerendsenThermostat, NVEThermostat, _LJ_force_and_energy_on_particles
from ljhouses.tools import get_lattice_initial_conditions

from ljhouses.pythonsims import simulate, compute_LJ_force_and_energy

from time import time

import numpy as np

N = 200_000
LJ_r = 6
LJ_e = 2
LJ_Rmax = 4*6
g = 0.1
v0 = 0
dt = 0.01

N_rounds = 500
N_steps_per_round = 10
max_samples = 10

x, v, a = get_lattice_initial_conditions(N, v0, LJ_r)
#thermostat = StochasticBerendsenThermostat(10.0, N)
thermostat = NVEThermostat()

start = time()
_LJ_force_and_energy_on_particles(x, LJ_r, LJ_e, LJ_Rmax)
end = time()

print("C++ API needed {0:4.2f} seconds".format(end-start))

start = time()
compute_LJ_force_and_energy(x, LJ_r, LJ_e, LJ_Rmax)
end = time()

print("Python API needed {0:4.2f} seconds".format(end-start))


#print(samples[-1][0][:2])
#print(samples[-2][0][:2])
#
#
#import matplotlib.pyplot as pl
#
#pl.figure()
#pl.plot(t, K)
#pl.plot(t, V)
#pl.plot(t, Vij)
#pl.plot(t, np.sum((K, V, Vij),axis=0))
#
#pl.show()
