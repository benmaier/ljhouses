import numpy as np
import matplotlib.pyplot as pl

from ljhouses.pythonsims import simulate_collisions_until_no_collisions_simple
from ljhouses.drawing import plot_configuration

from fincoretails import unipareto, powpareto

alpha = 4
xmin = 2


N = 1400
radiuses = powpareto.sample(N, alpha, xmin)

L = 10
R = L/np.sqrt(N)/np.pi * radiuses
#R = radiuses
x = np.random.rand(N,2) * L

from ljhouses.tools import get_ideal_gas_from_theory

x = get_ideal_gas_from_theory(N, L)


fig, ax = pl.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
ax[0].set_title('initial configuration')

plot_configuration(x, R, ax[0])
xnew = simulate_collisions_until_no_collisions_simple(x, R)
plot_configuration(xnew, R, ax[1])
ax[1].set_title('after running collision detection')

fig.tight_layout()
fig.savefig('erlang_collision_example_varying_radius.png', dpi=150)



pl.show()
