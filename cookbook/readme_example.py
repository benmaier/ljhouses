import numpy as np
import matplotlib.pyplot as pl

from ljhouses.pythonsims import get_close_to_equilibrium_initial_conditions, simulate, StochasticBerendsenThermostat
from ljhouses.drawing import plot_configuration

# define system parameters
N = 1000          # number of particles
LJ_r = 10         # potential minimum for interaction energy
LJ_e = 20         # potential depth
LJ_Rmax = 3*LJ_r  # maximum distance to which to compute interaction forces
g = 0.2           # gravitational pull to the center
v0 = 6            # root mean squared velocity of the particles (~root temperature)
dt = 0.01         # length of time step

# create display figure
fig, ax = pl.subplots(1, 3, figsize=(10,4))
ax[0].sharex(ax[1])
ax[0].sharey(ax[1])

# create initial conditions that should lie close to an equilibrium
# configuration and display the configuration
x, v, a = get_close_to_equilibrium_initial_conditions(N, v0, LJ_r, g, dt)
plot_configuration(x, LJ_r/2, ax[0])
ax[0].set_title('initial configuration')

# Define a thermostat that keeps the configuration at roughly the
# same temperature
thermostat = StochasticBerendsenThermostat(v0, N)

# define more simulation parameters
N_sampling_rounds = 10
N_steps_per_sample = 1000
max_samples = 10

# run the simulation
samples, t, K, V, Vij = simulate(dt,
                                 N_sampling_rounds,
                                 N_steps_per_sample,
                                 max_samples,
                                 LJ_r,
                                 LJ_e,
                                 LJ_Rmax,
                                 g,
                                 x,
                                 v,
                                 a,
                                 thermostat)

# obtain the final configuration and display it
x1, v1, a1 = samples[-1]
plot_configuration(x1, LJ_r/2, ax[1])
ax[1].set_title(f'after {int(np.round(t[-1]/dt)):d} steps')

# plot the temporal evolution of the energies
ax[2].plot(t, K, label='K')
ax[2].plot(t, V, label='V')
ax[2].plot(t, Vij, label='Vij')
ax[2].plot(t, np.sum((K, V, Vij),axis=0), label='E')

ax[2].set_xlabel('time (arb. units)')
ax[2].set_ylabel('energy (arb. units)')

ax[2].legend(frameon=True,fancybox=False)
fig.tight_layout()
fig.savefig('example_simulation.png',dpi=150)


pl.show()
