import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import EllipseCollection

from ljhouses.pythonsims import simulate_collisions_until_no_collisions

def plot_configuration(x,radius,ax=None):
    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=(10,10))
    ax.set_aspect('equal')
    D = 2*radius
    sizes = np.ones(N) * D
    coll = EllipseCollection(sizes, sizes, np.zeros_like(sizes), offsets=x, transOffset=ax.transData, alpha=0.2,units='x')
    ax.add_collection(coll)

N = 300
L = 1
R = L/np.sqrt(N)/np.pi
D = 2*R
x = np.random.rand(N,2) * L


fig, ax = pl.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

plot_configuration(x, R, ax[0])
xnew = simulate_collisions_until_no_collisions(x, D)
plot_configuration(xnew, R, ax[1])



pl.show()
