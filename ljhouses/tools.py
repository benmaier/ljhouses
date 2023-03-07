import numpy as np
from _ljhouses import StochasticBerendsenThermostat
from scipy.stats import gamma
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from numba import njit

def _initial_lattice(N, LJ_r):
    """
    Return N lattice positions as an approximate square
    with site distanace of LJ_r
    """
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

def get_ideal_gas_initial_conditions(N, root_mean_squared_velocity, g):
    v0 = root_mean_squared_velocity
    v = v0 * np.random.randn(N,2) / np.sqrt(2)
    x = get_ideal_gas_from_kinetic_gas_theory(N, root_mean_squared_velocity, g)
    a = np.zeros((N,2))
    return x, v, a

class NVEThermostat(StochasticBerendsenThermostat):
    """
    A thermostat that literally does nothing
    (and therefore leads to a simulation of the NVE ensemble)
    """

    def __init__(self,*args,**kwargs):
        super().__init__()

def get_ideal_gas_from_positions(pos,center=np.array([0.,0.])):
    """Randomly rotate positions around their center"""
    r = np.linalg.norm(pos-center[None,:],axis=1)
    return get_ideal_gas_from_radial_distances(r)

def get_ideal_gas_from_radial_distances(r):
    """Create positions at distance r with random angles"""
    theta = np.random.rand(r.shape[0])*2*np.pi
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return xy2pos(x, y)

def get_ideal_gas_from_theory(N, mean_distance_to_center):
    erlang_k = 2
    erlang_lambda = erlang_k/mean_distance_to_center
    r = gamma.rvs(a=erlang_k, scale=1/erlang_lambda, size=N)
    return get_ideal_gas_from_radial_distances(r)

def get_ideal_gas_from_kinetic_gas_theory(N,root_mean_squared_velocity,g):
    """
    Given a root mean squared velocity sqrt(<v^2>) and a gravitational
    constant, generate ideal gas positions.
    """
    v2 = root_mean_squared_velocity**2
    T = 0.5 * v2 # in 2d: T = K/N and K = 0.5 * N * <v^2>
    erlang_lambda = g/T
    erlang_k = 2
    mean_distance_to_center = erlang_k/erlang_lambda
    return get_ideal_gas_from_theory(N, mean_distance_to_center)

def pos2xy(pos):
    """Given one 2d-array of shape (N, 2), return two 1d-arrays of shape (N,)"""
    return pos[:,0], pos[:,1]

def xy2pos(x, y):
    """Given two 1d-arrays of shape (N,), return a 2d-array of shape (N, 2)"""
    return np.array([x,y]).T

def get_pairwise_distances_ball(pos,Rmax,T=None):
    """Return an (N,)-shaped array that contains distances of all pairs that lie within distance Rmax"""
    T_is_None = T is None
    if T_is_None:
        T = KDTree(pos)
    source_target_pairs = T.query_pairs(Rmax,output_type='ndarray')
    s = source_target_pairs[:,0]
    t = source_target_pairs[:,1]
    rv = pos[t,:] - pos[s,:]
    r = np.linalg.norm(rv,axis=1)
    if T_is_None:
        del T
    return r

def np_2d_add_at(subject,
              source_indices,
              target_indices,
              what_to_add
              ):
    """Add values to a 2d-array at `source_indices` and subtract the same at `target_indices`"""
    np.add.at(subject[:,0], source_indices, what_to_add[:,0])
    np.add.at(subject[:,0], target_indices, -what_to_add[:,0])
    np.add.at(subject[:,1], source_indices, what_to_add[:,1])
    np.add.at(subject[:,1], target_indices, -what_to_add[:,1])

@njit
def get_random_pairs(N, p):
    """
    Construct a G(N,p) random graph with method
    by Batagelj & Brandes (2005).
    """

    src = []
    trg = []

    u = 1
    v = -1
    logp = np.log(1.0 - p)
    while u < N:
        logr = np.log(1.0 - np.random.rand())
        v = v + 1 + int(np.floor(logr / logp))
        while v >= u and u < N:
            v = v - u
            u = u + 1
        if u < N:
            src.append(u)
            trg.append(v)

    return np.array(src), np.array(trg)

def get_sampled_pairwise_distances(pos,N_pair_samples):
    """
    Return an array that contains distances
    between subsampled pairs
    """
    N = pos.shape[0]
    p = N_pair_samples / (0.5*N*(N-1))
    s, t = get_random_pairs(N, p)
    rv = pos[t,:] - pos[s,:]
    return np.linalg.norm(rv)

if __name__ == "__main__":
    pos = np.random.rand(10, 2)
    print(get_sampled_pairwise_distances(pos,10))
    print(get_random_pairs(10,10/45))
