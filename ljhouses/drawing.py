"""
Contains a few matplotlib functions for drawing
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import EllipseCollection
from matplotlib.pyplot import Axes

def plot_configuration(x,radius,ax=None,facecolors='w',linewidths=1.5,edgecolors='#333333',alpha=0.5) -> Axes:
    """
    Plot a single 2D configuration of spheres with radius `radius`. Returns the axis that was used for drawing.
    """
    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=(6,6))
    ax.set_aspect('equal')
    D = 2*radius
    extent = (
                np.min(x[:,0]-radius),
                np.max(x[:,0]+radius),
                np.min(x[:,1]-radius),
                np.max(x[:,1]+radius),
            )
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    sizes = np.ones(x.shape[0]) * D
    coll = EllipseCollection(widths=sizes,
                             heights=sizes,
                             angles=np.zeros_like(sizes),
                             units='x',
                             offsets=x.copy(),
                             transOffset=ax.transData,
                             alpha=alpha,
                             facecolors=facecolors,
                             linewidths=linewidths,
                             edgecolors=edgecolors,
                             )
    ax.add_collection(coll)
    return ax

if __name__=="__main__":

    N = 300
    L = 10
    R = L/np.sqrt(N)/np.pi
    D = 2*R
    x = np.random.rand(N,2) * L
    plot_configuration(x, R,)

    pl.show()
