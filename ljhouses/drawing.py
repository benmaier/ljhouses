"""
Contains a few matplotlib functions for drawing
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.pyplot import Axes
from scipy.spatial import Delaunay

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

def plot_delaunay(pos, ax=None,color='#555555'):

    N = pos.shape[0]
    tri = Delaunay(pos)
    source = []
    target = []
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i+1) % 3]
            source.append(u)
            target.append(v)

    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=figsize)
    _ax = pl.gca()
    pl.sca(ax)
    pl.axis('square')
    pl.axis('off')
    pl.sca(_ax)

    pos0 = pos[source,:]
    pos1 = pos[target,:]

    lines = list(zip(pos0, pos1))
    coll = LineCollection(lines,linewidths=0.5,colors=color)
    ax.add_collection(coll)
    #ax.plot(pos[:,0], pos[:,1],'.',zorder=-1000,alpha=0)

    return ax

def plot_links(x,sources,targets,ax=None,linewidths=1.5,edgecolors='#333333',alpha=0.5) -> Axes:
    """
    Plot a single 2D configuration of spheres with radius `radius`. Returns the axis that was used for drawing.
    """
    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=(6,6))
    ax.set_aspect('equal')
    extent = (
                (x[:,0]).min(),
                (x[:,0]).max(),
                (x[:,1]).min(),
                (x[:,1]).max(),
            )

    lines = [ \
             ( x[s,:], x[t,:] ) \
              for s, t in zip(sources, targets)
            ]

    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    coll = LineCollection(lines,
                          alpha=alpha,
                          linewidths=linewidths,
                          colors=edgecolors,
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
