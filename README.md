# ljhouses

Simulate Lennard-Jones spheres and similar systems in the context of location of buildings.

## Install

    git clone https://github.com/benmaier/ljhouses/
    pip install ./ljhouses

`ljhouses` was developed and tested for 

* Python 3.10
* Python 3.11

So far, the package's functionality was tested on macOS X and Ubuntu only.

## Dependencies

`ljhouses` directly depends on the following packages which will be installed by `pip` during the installation process

* `numpy>=1.17`

## Examples

### Thermalized LJ simulation


```python
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
```

![Example simulation](https://github.com/benmaier/ljhouses/blob/main/cookbook/example_simulation.png)

### Collision simulation


```python
import numpy as np
import matplotlib.pyplot as pl

from ljhouses.pythonsims import simulate_collisions_until_no_collisions
from ljhouses.drawing import plot_configuration

N = 300
L = 1
R = L/np.sqrt(N)/np.pi
D = 2*R
x = np.random.rand(N,2) * L

fig, ax = pl.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
ax[0].set_title('initial configuration')

plot_configuration(x, R, ax[0])
xnew = simulate_collisions_until_no_collisions(x, D)
plot_configuration(xnew, R, ax[1])
ax[1].set_title('after running collision detection')
```

![Collision simulation](https://github.com/benmaier/ljhouses/blob/main/cookbook/collision/collision_example.png?raw=true)
                        

## Changelog

Changes are logged in a [separate file](https://github.com/benmaier/ljhouses/blob/main/CHANGELOG.md).

## License

Most of this project is licensed under the [BSD-3 License](https://github.com/benmaier/ljhouses/blob/main/LICENSE).
Note that this excludes any images/pictures/figures shown here or in the documentation.

## Contributing

If you want to contribute to this project, please make sure to read the [code of conduct](https://github.com/benmaier/ljhouses/blob/main/CODE_OF_CONDUCT.md) and the [contributing guidelines](https://github.com/benmaier/ljhouses/blob/main/CONTRIBUTING.md). In case you're wondering about what to contribute, we're always collecting ideas of what we want to implement next in the [outlook notes](https://github.com/benmaier/ljhouses/blob/main/OUTLOOK.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code-of-conduct.md)

## Dev notes

Fork this repository, clone it, and install it in dev mode.

```bash
git clone git@github.com:YOURUSERNAME/ljhouses.git
make
```

If you want to upload to PyPI, first convert the new `README.md` to `README.rst`

```bash
make readme
```

It will give you warnings about bad `.rst`-syntax. Fix those errors in `README.rst`. Then wrap the whole thing 

```bash
make pypi
```

It will probably give you more warnings about `.rst`-syntax. Fix those until the warnings disappear. Then do

```bash
make upload
```
