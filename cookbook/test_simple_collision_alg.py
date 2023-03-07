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

fig.tight_layout()
fig.savefig('collision_example.png', dpi=150)



pl.show()
