"""
Simulation functions in pure python.
"""
import numpy as np
from scipy.spatial import KDTree
from numpy.typing import NDArray
from _ljhouses import StochasticBerendsenThermostat
from ljhouses.tools import NVEThermostat, get_ideal_gas_initial_conditions, np_2d_add_at
from tqdm import tqdm

Arr = NDArray[np.float64]
IArr = NDArray[np.int64]
f64 = np.float64
Samples = list[tuple[Arr,Arr,Arr]]

def total_kinetic_energy(v: Arr) -> f64:
    return 0.5*(v**2).sum()

def total_interaction_energy(x: Arr, LJ_r: float, LJ_e: float, LJ_Rmax: float) -> f64:
    return np.sum(compute_LJ_force_and_energy(x, LJ_r, LJ_e, LJ_Rmax)[1])

def total_potential_energy(x: Arr, g: float) -> f64:
    return np.sum(compute_gravitational_force_and_energy(x, g)[1])

def compute_LJ_force(xi: Arr,
                     xj: Arr,
                     LJ_R2: float,
                     LJ_energy: float,
                     ) -> Arr:
    rv = xj - xi
    rSq = rv.dot(rv)
    r2 = LJ_R2 / rSq
    r6 = r2**3
    r12 = r6**2
    return -12*rv/rSq * LJ_energy * (r12-r6)

def compute_LJ_force_arr(rv: Arr,
                         rSq: Arr,
                         LJ_R2: float,
                         LJ_energy: float,
                         ) -> Arr:
    r2 = LJ_R2/rSq
    r6 = r2**3
    r12 = r6**2
    return -12 * rv * LJ_energy * np.expand_dims((r12-r6)/rSq,-1)

def compute_LJ_energy(rSq: float | Arr,
                      LJ_R2: float,
                      LJ_energy: float,
                      ) -> float | Arr:
    r2 = LJ_R2 / rSq
    r6 = r2**3
    r12 = r6**2
    return LJ_energy * (r12-2*r6)

def compute_LJ_force_and_energy(pos: Arr,
                                LJ_r: float,
                                LJ_e: float,
                                LJ_Rmax: float
                                ) -> tuple[Arr, Arr]:

    forces = np.zeros_like(pos)
    energies = np.zeros_like(pos[:,0])
    if LJ_e == 0.0:
        return forces, energies

    LJ_R2 = LJ_r**2
    T = KDTree(pos)
    offset = compute_LJ_energy(LJ_Rmax**2, LJ_R2, LJ_e)
    source_target_pairs = T.query_pairs(LJ_Rmax,output_type='ndarray')
    s = source_target_pairs[:,0]
    t = source_target_pairs[:,1]
    rv = pos[t,:] - pos[s,:]
    rSq = np.sum(rv**2,axis=1)
    F = compute_LJ_force_arr(rv, rSq, LJ_R2, LJ_e)
    V = compute_LJ_energy(rSq, LJ_R2, LJ_e) - offset
    np_2d_add_at(forces, s, t, F)
    np.add.at(energies, s, 0.5*V)
    np.add.at(energies, t, 0.5*V)
    return forces, energies

def compute_gravitational_force_and_energy(pos: Arr,
                                          g: float,
                                         ) -> tuple[Arr,Arr]:
    r = np.linalg.norm(pos,axis=1)
    return (
                - pos / r[:,None] * g,
                r*g,
            )

def update_verlet(x: Arr,
                  v: Arr,
                  a: Arr,
                  dt: float,
                  LJ_r: float,
                  LJ_e: float,
                  LJ_Rmax: float,
                  g: float,
                ) -> tuple[Arr,Arr,Arr,f64,f64,f64]:

    x += v * dt + a * 0.5*dt*dt

    anew, pot_energy = compute_gravitational_force_and_energy(x, g)

    LJ_force, LJ_energy = compute_LJ_force_and_energy(x, LJ_r, LJ_e, LJ_Rmax)
    anew += LJ_force

    v += (anew + a) * 0.5 * dt

    a = anew

    return (x, v, a, total_kinetic_energy(v), pot_energy.sum(), LJ_energy.sum())

def simulate(
        dt: float,
        N_sampling_rounds: int,
        N_steps_per_sample: int,
        max_samples: int,
        LJ_r: float,
        LJ_e: float,
        LJ_Rmax: float,
        g: float,
        positions: Arr,
        velocities: Arr,
        accelerations: Arr,
        thermostat: StochasticBerendsenThermostat | NVEThermostat = None,
        show_progress : bool = False,
    ) -> tuple[Samples, Arr, Arr, Arr, Arr]:
    """
    Run an MD simulation of LJ spheres in a linear potential,
    returning configuration samples and respective time series
    for all energies.

    Parameters
    ==========
    dt : float
        Time advancement per step
    N_sampling_rounds : int
        How many consecutive configuration sampling rounds to run
    N_steps_per_sample : int
        How many time steps to run per sampling round
    max_samples : int
        Maximum number of samples to save and return
    LJ_r : float
        Where the potential minimum lies in the pair-wise interaction
        potential (can be thought of a single sphere's effective diameter)
    LJ_e : float
        Potential well depth in relation to gauge energy Vij = 0.
    LJ_Rmax : float
        Interaction forces and energies will not be computed for pairs
        of spheres that lie farther than this distance
    g : float
        gravitational constant of linear gravitational potential
    positions : numpy.ndarry
        initial positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    velocities : numpy.ndarry
        initial velocities of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    accelerations : numpy.ndarry
        initial accelerations of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    thermostat : StochasticBerendsenThermostat | NVEThermostat, default = None
        The thermostat to use to equilibrate. Passing `None` or `NVEThermostat`
        will both result in simulating an NVE ensemble
    show_progress : bool default = False
        Show progress bar with simulation state

    Returns
    =======
    samples : list of tuples
        A list containing the last samples that have been taken.
        Each sample is a tuple of format

        .. code:: python

            (positions, velocities, accelerations)

    t : numpy.ndarray
        An array that contains every individual time step
    K : numpy.ndarray
        An array that contains the total kinetic energy at
        every individual time step
    V : numpy.ndarray
        An array that contains the total gravitational energy at
        every individual time step
    Vij : numpy.ndarray
        An array that contains the total interaction energy at
        every individual time step
    """

    thermostat_is_active = thermostat is not None and thermostat.is_active

    x = positions
    v = velocities
    a = accelerations

    samples = []

    t = 0.0
    time = [t]
    kinetic_energy = [total_kinetic_energy(velocities)]
    potential_energy = [np.sum(compute_gravitational_force_and_energy(positions, g)[1])]
    interaction_energy = [np.sum(compute_LJ_force_and_energy(positions, LJ_r, LJ_e, LJ_Rmax)[1])]

    if show_progress:
        bar = tqdm(total=N_sampling_rounds*N_steps_per_sample)

    for sample in range(N_sampling_rounds):
        for step in range(N_steps_per_sample):
            x, v, a, K, V, Vij = update_verlet(x, v, a, dt, LJ_r, LJ_e, LJ_Rmax, g)
            if thermostat_is_active:
                v = np.array(thermostat.get_thermalized_velocities(v, K))
                K = total_kinetic_energy(v)
            t += dt
            time.append(t)
            kinetic_energy.append(K)
            potential_energy.append(V)
            interaction_energy.append(Vij)

            if show_progress:
                bar.update()

        samples.append((x.copy(),v.copy(),a.copy()))
        if len(samples) > max_samples:
            samples = samples[1:]

    return (
               samples,
               np.array(time),
               np.array(kinetic_energy),
               np.array(potential_energy),
               np.array(interaction_energy)
           )

def simulate_once(
        positions: Arr,
        velocities: Arr,
        accelerations: Arr,
        dt: float,
        LJ_r: float,
        LJ_e: float,
        LJ_Rmax: float,
        g: float,
        Nsteps: int,
        thermostat: StochasticBerendsenThermostat | NVEThermostat = None,
    ) -> tuple[Arr, Arr, Arr, float, float, float]:
    """
    Run an MD simulation of LJ spheres in a linear potential, once.

    Parameters
    ==========
    positions : numpy.ndarry
        initial positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    velocities : numpy.ndarry
        initial velocities of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    accelerations : numpy.ndarry
        initial accelerations of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    dt : float
        Time advancement per step
    max_samples : int
        Maximum number of samples to save and return
    LJ_r : float
        Where the potential minimum lies in the pair-wise interaction
        potential (can be thought of a single sphere's effective diameter)
    LJ_e : float
        Potential well depth in relation to gauge energy Vij = 0.
    LJ_Rmax : float
        Interaction forces and energies will not be computed for pairs
        of spheres that lie farther than this distance
    g : float
        gravitational constant of linear gravitational potential
    N_steps_per_sample : int
        How many time steps to run
    thermostat : StochasticBerendsenThermostat | NVEThermostat, default = None
        The thermostat to use to equilibrate. Passing `None` or `NVEThermostat`
        will both result in simulating an NVE ensemble

    Returns
    =======
    positions : numpy.ndarry
        final positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    velocities : numpy.ndarry
        final velocities of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    accelerations : numpy.ndarry
        final accelerations of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    K : float
        final total kinetic energy
    V : float
        final total gravitational energy
    Vij : numpy.ndarray
        final total interaction energy
    """

    thermostat_is_active = thermostat is not None and thermostat.is_active

    x = positions
    v = velocities
    a = accelerations

    for step in range(Nsteps):
        x, v, a, K, V, Vij = update_verlet(x, v, a, dt, LJ_r, LJ_e, LJ_Rmax, g)
        if thermostat_is_active:
            v = np.array(thermostat.get_thermalized_velocities(v, K))
            K = total_kinetic_energy(v)

    return x, v, a, K, V, Vij


def _pair(a,b):
    if a > b:
        return b, a
    elif a < b:
        return a, b
    else:
        return None

def update_collisions_varying_radius(
                      x: Arr,
                      radiuses: Arr,
                      collision_strength: float = 1.0,
                      eps: float = 1e-10,
                      attract_within_diameter: bool = False,
                      v: Arr = None,
                      a: Arr = None,
                    ):

    # construct KDTree and compute the distance
    # for all pairs that lie within distance LJ_r
    T = KDTree(x)

    N = x.shape[0]
    source_target_pairs = set()
    for i in range(N):
        neighbors = T.query_ball_point(x[i], 2*radiuses[i])
        for j in neighbors:
            this_pair = _pair(i,j)
            if this_pair is not None:
                source_target_pairs.add(this_pair)

    if len(source_target_pairs) == 0:
        return None

    source_target_pairs = np.array(list(source_target_pairs),dtype=int)
    s = source_target_pairs[:,0]
    t = source_target_pairs[:,1]
    rv = x[s,:] - x[t,:]
    r = np.linalg.norm(rv,axis=1)

    LJ_r = radiuses[s] + radiuses[t]

    overlap_indices = np.where(r < LJ_r)[0]
    if len(overlap_indices) == 0 and not attract_within_diameter:
        return None
    elif len(overlap_indices) > 0 and not attract_within_diameter:
        s = s[overlap_indices]
        t = t[overlap_indices]
        rv = rv[overlap_indices,:]
        r = r[overlap_indices]
        LJ_r = LJ_r[overlap_indices]

    # compute the amount that this single interaction should make the sphere should be move
    D = rv * ((LJ_r-r)/2/r)[:,None]

    # scale by collision strength
    D *= collision_strength

    # For each sphere, compute a Delta vector that says in which direction the sphere should move
    # from summing up all the individual contributions
    DELTA = np.zeros_like(x)
    np_2d_add_at(DELTA, s, t, D)

    # Threshold this vector by demanding that the change is not more than a diameter of the speheres 
    rDELTA = np.linalg.norm(DELTA,axis=1)
    ind = np.where(rDELTA>2*radiuses)[0]
    DELTA[ind,:] = DELTA[ind,:]/rDELTA[ind,None] * 2*radiuses[ind,None]
    rDELTA[ind] = 2*radiuses[ind]

    if not np.any(rDELTA>eps):
        return None

    # add the Delta vector to the positions
    x += DELTA

    if v is not None or a is not None:
        # filter by which spheres actually need to change their position
        not_changed = np.where(rDELTA==0.0)[0]
        changed = np.where(rDELTA>0.0)[0]

        # norm the the Delta vector for spheres that changed
        rDELTA[not_changed] = 1.0
        DELTANORMED = DELTA / rDELTA[:,None]

        # for spheres that have changed position,
        # reflect velocity and acceleration according to the
        # corresponding DELTA vector

        if v is not None:
            vnorm = np.linalg.norm(v, axis=1)
            v[changed,:] = DELTANORMED[changed,:] * vnorm[changed,None]

        if a is not None:
            anorm = np.linalg.norm(a, axis=1)
            a[changed,:] = DELTANORMED[changed,:] * anorm[changed,None]

    # return the tree
    return T

def update_collisions(x: Arr,
                      v: Arr,
                      a: Arr,
                      LJ_r: float,
                      collision_strength: float = 1.0,
                      ):

    # construct KDTree and compute the distance
    # for all pairs that lie within distance LJ_r
    T = KDTree(x)
    source_target_pairs = T.query_pairs(LJ_r,output_type='ndarray')
    s = source_target_pairs[:,0]
    t = source_target_pairs[:,1]
    rv = x[s,:] - x[t,:]
    r = np.linalg.norm(rv,axis=1)

    # compute the amount that this single interaction should make the sphere should be move
    D = rv * ((LJ_r-r)/2/r)[:,None]

    # scale by collision strength
    D *= collision_strength

    # For each sphere, compute a Delta vector that says in which direction the sphere should move
    # from summing up all the individual contributions
    DELTA = np.zeros_like(x)
    np_2d_add_at(DELTA, s, t, D)

    # Threshold this vector by demanding that the change is not more than a diameter of the speheres 
    rDELTA = np.linalg.norm(DELTA,axis=1)
    ind = np.where(rDELTA>LJ_r)[0]
    DELTA[ind,:] = DELTA[ind,:]/rDELTA[ind,None] * LJ_r
    rDELTA[ind] = LJ_r

    # add the Delta vector to the positions
    x += DELTA

    # filter by which spheres actually need to change their position
    not_changed = np.where(rDELTA==0.0)[0]
    changed = np.where(rDELTA>0.0)[0]

    # norm the the Delta vector for spheres that changed
    rDELTA[not_changed] = 1.0
    DELTANORMED = DELTA / rDELTA[:,None]

    # for spheres that have changed position,
    # reflect velocity and acceleration according to the
    # corresponding DELTA vector
    vnorm = np.linalg.norm(v, axis=1)
    anorm = np.linalg.norm(a, axis=1)
    v[changed,:] = DELTANORMED[changed,:] * vnorm[changed,None]
    a[changed,:] = DELTANORMED[changed,:] * anorm[changed,None]

    # return the tree
    return T

def simulate_collisions_once(
        positions: Arr,
        velocities: Arr,
        accelerations: Arr,
        dt: float,
        LJ_r: float,
        LJ_e: float,
        LJ_Rmax: float,
        g: float,
        Nsteps: int,
        *args,
        collision_strength: float = 1.0,
        thermostat: StochasticBerendsenThermostat | NVEThermostat = None,
        **kwargs,
    ) -> tuple[Arr, Arr, Arr, float, float, float]:
    """
    Run a collision simulation of LJ spheres, once.

    Parameters
    ==========
    positions : numpy.ndarry
        initial positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    velocities : numpy.ndarry
        initial velocities of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    accelerations : numpy.ndarry
        initial accelerations of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    dt : float
        Time advancement per step
    max_samples : int
        Maximum number of samples to save and return
    LJ_r : float
        Where the potential minimum lies in the pair-wise interaction
        potential (can be thought of a single sphere's effective diameter)
    LJ_e : float
        Potential well depth in relation to gauge energy Vij = 0.
    LJ_Rmax : float
        Interaction forces and energies will not be computed for pairs
        of spheres that lie farther than this distance
    g : float
        gravitational constant of linear gravitational potential
    N_steps_per_sample : int
        How many time steps to run
    collision_strength : float, default = 1.0
        This is the scalar that the collision-vector will be scaled with
        at each timestep
    thermostat : StochasticBerendsenThermostat | NVEThermostat, default = None
        The thermostat to use to equilibrate. Passing `None` or `NVEThermostat`
        will both result in simulating an NVE ensemble

    Returns
    =======
    positions : numpy.ndarry
        final positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    velocities : numpy.ndarry
        final velocities of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    accelerations : numpy.ndarry
        final accelerations of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    K : float
        final total kinetic energy
    V : float
        final total gravitational energy
    Vij : numpy.ndarray
        final total interaction energy
    """
    thermostat_is_active = thermostat is not None and thermostat.is_active

    x = positions
    v = velocities
    a = accelerations

    K = 0
    for step in range(Nsteps):
        update_collisions(x, v, a, LJ_r, collision_strength=collision_strength)
        x, v, a, K, V, Vij = update_verlet(x, v, a, dt, LJ_r, LJ_e=0, LJ_Rmax=0, g=0)
        if thermostat_is_active:
            #K = total_kinetic_energy(v)
            v = np.array(thermostat.get_thermalized_velocities(v, K))
            K = total_kinetic_energy(v)

    Vij = total_interaction_energy(x, LJ_r, LJ_e, LJ_Rmax)
    V = total_potential_energy(x, g)

    return x, v, a, K, V, Vij


def get_close_to_equilibrium_initial_conditions(
        N : int,
        root_mean_squared_velocity: float,
        LJ_r: float,
        g: float,
        dt: float,
        cutoff_factor: float = 0.98,
        *args,
        **kwargs,
    ) -> tuple[Arr, Arr, Arr]:
    """
    Initiate an ideal gas from kinetic gas theory.
    Then run a collision simulation of LJ spheres of radius ``LJ_r/2``
    until there's no pairs of spheres left that are within
    distance < cutoff_factor*LJ_r.
    Then generate velocities according to Maxwell-Boltzmann distribution
    and set accelerations to zero.

    Parameters
    ==========
    N : int
        number of spheres
    root_mean_squared_velocity: float
        ``sqrt(<v^2)``
    LJ_r : float
        Where the potential minimum lies in the pair-wise interaction
        potential (can be thought of a single sphere's effective diameter)
    LJ_e : float
        Potential well depth in relation to gauge energy Vij = 0.
    LJ_Rmax : float
        Interaction forces and energies will not be computed for pairs
        of spheres that lie farther than this distance
    g : float
        gravitational constant of linear gravitational potential
    dt : float
        Time advancement per step
    cutoff_factor : float or str, default = 0.98
        When to consider two particles to be overlapping. The cutoff
        is calculated as ``cutoff_factor * LJ_R``. The default
        cutoff factor will lead to a distance (diameter) that is slightly
        below the minimum of the potential (98%). To achieve the
        Lennard-Jones ``sigma`` parameter, i.e. the zero of the potential
        instead, pass ``cutoff_factor = 0.891`` or
        ``cutoff_factor=1/2**(1/6)``.

    Returns
    =======
    positions : numpy.ndarry
        final positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    velocities : numpy.ndarry
        final velocities of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    accelerations : numpy.ndarry
        final accelerations of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    """

    cutoff = cutoff_factor * LJ_r

    x, v, a = get_ideal_gas_initial_conditions(N, root_mean_squared_velocity, g)

    v0 = v.copy()
    a0 = a.copy()

    thermostat = StochasticBerendsenThermostat(root_mean_squared_velocity, N)

    T = KDTree(x)
    pairs = T.query_pairs(cutoff,output_type='ndarray')
    while pairs.shape[0] != 0:

        T = update_collisions(x, v, a, LJ_r)
        x, v, a, K, V, Vij = update_verlet(x, v, a, dt, LJ_r, LJ_e=0, LJ_Rmax=0, g=0)

        K = total_kinetic_energy(v)
        v = np.array(thermostat.get_thermalized_velocities(v, K))
        K = total_kinetic_energy(v)

        pairs = T.query_pairs(cutoff,output_type='ndarray')

    return x, v0, a0



def simulate_collisions_until_no_collisions(
        positions:  Arr,
        distance_cutoff: float,
        root_mean_squared_velocity: float = 0,
        dt: float = 0.01,
        collision_strength: float = 1.0,
        *args,
        releps: float = 1e-3,
        **kwargs,
    ) -> Arr:
    """
    Simulate N hard spheres of radius `distance_cutoff/2` until
    there's no overlap anymore.

    Parameters
    ==========
    positions : numpy.ndarry
        initial positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    distance_cutoff : float
        if the centers of two sphere are below this distance
        we consider this as a collision
    root_mean_squared_velocity: float
        ``sqrt(<v^2>)``, if >0, we use this to support the
        equilibration process using verlet integration
    dt : float, default = 0.01
        Time advancement per step, only used for the verlet
        integration
    collision_strength : float, default = 1.0
        This is the scalar that the collision-vector will be scaled with
        at each timestep
    releps : float, default = 1e-3
        relative tolerance in overlap (allow overlap that is ~(1-releps)*distance_cutoff)

    Returns
    =======
    positions : numpy.ndarry
        final positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    """

    x = positions.copy()

    N = x.shape[0]

    v = np.random.randn(*x.shape) / np.sqrt(2) * root_mean_squared_velocity
    a = np.zeros_like(v)

    thermostat = StochasticBerendsenThermostat(root_mean_squared_velocity, N)

    scl = 1-releps
    T = KDTree(x)
    pairs = T.query_pairs(distance_cutoff*scl,output_type='ndarray')
    while pairs.shape[0] != 0:

        T = update_collisions(x, v, a, distance_cutoff, collision_strength=collision_strength)
        if root_mean_squared_velocity > 0:
            x, v, a, K, V, Vij = update_verlet(x, v, a, dt, distance_cutoff, LJ_e=0, LJ_Rmax=0, g=0)

            K = total_kinetic_energy(v)
            v = np.array(thermostat.get_thermalized_velocities(v, K))
            K = total_kinetic_energy(v)

        pairs = T.query_pairs(distance_cutoff*scl,output_type='ndarray')

    return x




def simulate_collisions_until_no_collisions_simple(
        positions:  Arr,
        radiuses:  Arr,
        #distance_cutoff: float,
        #collision_strength: float = 1.0,
        eps: float = 1e-10,
        attract_within_diameter: bool = False,
        *args,
        **kwargs,
    ) -> Arr:
    """
    Simulate N hard spheres of radius `distance_cutoff/2` until
    there's no overlap anymore.

    Parameters
    ==========
    positions : numpy.ndarry
        initial positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    distance_cutoff : float
        if the centers of two sphere are below this distance
        we consider this as a collision
    root_mean_squared_velocity: float
        ``sqrt(<v^2>)``, if >0, we use this to support the
        equilibration process using verlet integration
    dt : float, default = 0.01
        Time advancement per step, only used for the verlet
        integration
    collision_strength : float, default = 1.0
        This is the scalar that the collision-vector will be scaled with
        at each timestep
    releps : float, default = 1e-3
        relative tolerance in overlap (allow overlap that is ~(1-releps)*distance_cutoff)

    Returns
    =======
    positions : numpy.ndarry
        final positions of the spheres.
        two-dimensional, ``shape = (N, dim)`` where ``N`` is the number
        of spheres and ``dim`` is the dimensionality of the problem
    """

    x = positions.copy()

    N = x.shape[0]

    while True:

        result = update_collisions_varying_radius(x, radiuses, attract_within_diameter=attract_within_diameter,eps=eps)

        if result is None:
            break

    return x




