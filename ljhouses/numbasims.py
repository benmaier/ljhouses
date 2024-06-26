"""
Simulation functions based on numba, mainly
for testing purposes.
"""
import numpy as np
from scipy.spatial import KDTree
from numpy.typing import NDArray
from _ljhouses import StochasticBerendsenThermostat
from ljhouses import NVEThermostat
from numba import njit, jit, typeof
from numba.core import types

from ljhouses.pythonsims import (
            Arr, f64, Samples,
            compute_gravitational_force_and_energy,
            compute_LJ_energy,
            compute_LJ_force,
            compute_LJ_force_and_energy,
            compute_LJ_force_arr,
            update_verlet,
            simulate,
            simulate_once,
            total_kinetic_energy,
        )

_2d = np.zeros((10,2))
_1d = np.zeros((10,))
nbArr2 = typeof(_2d)
nbArr1 = typeof(_1d)
nbf64 = types.float64

#@njit
#def total_kinetic_energy(v: Arr) -> f64:
#    s = 0.0
#    for _v in v.flatten():
#        s += _v*_v
#    return 0.5 * s

total_kinetic_energy = njit(total_kinetic_energy)



#compute_LJ_force = njit(compute_LJ_force, nbArr1(nbArr1, nbArr1, nbf64, nbf64))
compute_LJ_force = njit(compute_LJ_force)

compute_LJ_force_arr = njit(compute_LJ_force_arr)


if __name__=="__main__":
    v = np.random.rand(4,2)
    print(total_kinetic_energy(v))

    x = np.random.rand(4,2)
    compute_LJ_force(x[0],x[1], 2.0, 3.0)

    x_ = np.random.rand(4,2)
    rv = x_ - x
    rSq = ((x_ - x)**2).sum(axis=1)
    compute_LJ_force_arr(rv,rSq, 2.0, 3.0)

#@jit
#def compute_LJ_energy(rSq: float | Arr,
#                      LJ_R2: float,
#                      LJ_energy: float,
#                      ) -> float | Arr:
#    r2 = LJ_R2 / rSq
#    r6 = r2**3
#    r12 = r6**2
#    return LJ_energy * (r12-2*r6)
#
#@jit
#def compute_LJ_force_and_energy(pos: Arr,
#                                LJ_r: float,
#                                LJ_e: float,
#                                LJ_Rmax: float
#                                ) -> tuple[Arr, Arr]:
#    LJ_R2 = LJ_r**2
#    T = KDTree(pos)
#    forces = np.zeros_like(pos)
#    energies = np.zeros_like(pos[:,0])
#    if LJ_e == 0:
#        return forces, energies
#    offset = compute_LJ_energy(LJ_Rmax**2, LJ_R2, LJ_e)
#    source_target_pairs = T.query_pairs(LJ_Rmax,output_type='ndarray')
#    s = source_target_pairs[:,0]
#    t = source_target_pairs[:,1]
#    rv = pos[t,:] - pos[s,:]
#    rSq = np.sum(rv**2,axis=1)
#    F = compute_LJ_force_arr(rv, rSq, LJ_R2, LJ_e)
#    V = compute_LJ_energy(rSq, LJ_R2, LJ_e) - offset
#    anew_0 = forces[:,0]
#    anew_1 = forces[:,1]
#    np.add.at(anew_0, s, F[:,0])
#    np.add.at(anew_0, t, -F[:,0])
#    np.add.at(anew_1, s, F[:,1])
#    np.add.at(anew_1, t, -F[:,1])
#    np.add.at(energies, s, 0.5*V)
#    np.add.at(energies, t, 0.5*V)
#    return forces, energies
#
#@jit
#def compute_gravitational_force_and_energy(pos: Arr,
#                                          g: float,
#                                         ) -> tuple[Arr,Arr]:
#    r = np.linalg.norm(pos,axis=1)
#    return (
#                - pos / r[:,None] * g,
#                r*g,
#            )
#
#@jit
#def update_verlet(x: Arr,
#                  v: Arr,
#                  a: Arr,
#                  dt: float,
#                  LJ_r: float,
#                  LJ_e: float,
#                  LJ_Rmax: float,
#                  g: float,
#                ) -> tuple[Arr,Arr,Arr,f64,f64,f64]:
#
#    x += v * dt + a * 0.5*dt*dt
#
#    anew, pot_energy = compute_gravitational_force_and_energy(x, g)
#
#    LJ_force, LJ_energy = compute_LJ_force_and_energy(x, LJ_r, LJ_e, LJ_Rmax)
#    anew += LJ_force
#
#    v += (anew + a) * 0.5 * dt
#
#    a = anew
#
#    return (x, v, a, total_kinetic_energy(v), pot_energy.sum(), LJ_energy.sum())
#
#@jit
#def simulate(
#        dt: float,
#        N_sampling_rounds: int,
#        N_steps_per_sample: int,
#        max_samples: int,
#        LJ_r: float,
#        LJ_e: float,
#        LJ_Rmax: float,
#        g: float,
#        positions: Arr,
#        velocities: Arr,
#        accelerations: Arr,
#    ) -> tuple[Samples, Arr, Arr, Arr, Arr]:
#
#    x = positions
#    v = velocities
#    a = accelerations
#
#    samples = []
#
#    t = 0.0
#    time = [t]
#    kinetic_energy = [total_kinetic_energy(velocities)]
#    potential_energy = [compute_gravitational_force_and_energy(positions, g)[1]]
#    interaction_energy = [compute_LJ_force_and_energy(positions, LJ_r, LJ_e, LJ_Rmax)[1]]
#
#    for sample in range(N_sampling_rounds):
#        for step in range(N_steps_per_sample):
#            x, v, a, K, V, Vij = update_verlet(x, v, a, dt, LJ_r, LJ_e, LJ_Rmax, g)
#            t += dt
#            time.append(t)
#            kinetic_energy.append(K)
#            potential_energy.append(V)
#            interaction_energy.append(Vij)
#        samples.append((x,v,a))
#        if len(samples) > max_samples:
#            samples = samples[1:]
#
#    return samples, time, kinetic_energy, potential_energy, interaction_energy
#
#@jit
#def simulate_once(
#        positions: Arr,
#        velocities: Arr,
#        accelerations: Arr,
#        dt: float,
#        LJ_r: float,
#        LJ_e: float,
#        LJ_Rmax: float,
#        g: float,
#        Nsteps: int,
#        thermostat: StochasticBerendsenThermostat | NVEThermostat,
#    ) -> tuple[Arr, Arr, Arr, float, float, float]:
#
#    x = positions
#    v = velocities
#    a = accelerations
#
#    for step in range(Nsteps):
#        x, v, a, K, V, Vij = update_verlet(x, v, a, dt, LJ_r, LJ_e, LJ_Rmax, g)
#        if thermostat.is_active:
#            v = np.array(thermostat.get_thermalized_velocities(v, K))
#            K = total_kinetic_energy(v)
#
#    return x, v, a, K, V, Vij
#
#
#
