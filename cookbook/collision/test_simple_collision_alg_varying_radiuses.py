import numpy as np
import matplotlib.pyplot as pl

from ljhouses.pythonsims import simulate_collisions_until_no_collisions_simple
from ljhouses.drawing import plot_configuration

from fincoretails import unipareto

alpha = 3.33
xmin = 0.4


N = 1400
radiuses = unipareto.sample(N, alpha, xmin)

L = 1
R = L/np.sqrt(N)/np.pi * radiuses
x = np.random.rand(N,2) * L


fig, ax = pl.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
ax[0].set_title('initial configuration')

plot_configuration(x, R, ax[0])
xnew = simulate_collisions_until_no_collisions_simple(x, R)
plot_configuration(xnew, R, ax[1])
ax[1].set_title('after running collision detection')

fig.tight_layout()
fig.savefig('collision_example_varying_radius.png', dpi=150)



pl.show()
