import numpy as np

def _initial_lattice(N, LJ_r):
    L = int(np.floor(np.sqrt(N))+1)
    x = np.zeros((N, 2))

    for i in range(N):
        j = i % L
        k = i // L
        x[i, :] = [ (j-L/2)*LJ_r*2,
                    (k-L/2)*LJ_r*2
                  ]
    return x

def get_lattice_initial_conditions(N, root_mean_squared_velocity, LJ_r):
    """
    Return N particles on a lattice with velocity vectors
    that sum up to a certain root_mean_squared_velocity

    Parameters
    ==========
    N : int
        number of particles
    root_mean_squared_velocity : numpy.ndarray(N, 2)
        sqrt(E[|v|^2]) -- a measure for the initial kinetic energy
    LJ_r : float
        typical distance between the center of two houses

    Returns
    =======
    x : numpy.ndarray(N, 2)
        positions
    v : numpy.ndarray(N, 2)
        velocities
    a : numpy.ndarray(N, 2)
        accelerations, all equal to zero
    """

    v0 = root_mean_squared_velocity
    v = v0 * np.random.randn(N,2) / np.sqrt(2)
    x = _initial_lattice(N, LJ_r)
    a = np.zeros((N,2))

    return x, v, a
