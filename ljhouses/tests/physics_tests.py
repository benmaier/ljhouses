import unittest

import numpy as np

from ljhouses import (
        _LJ_force_and_energy,
        _LJ_force_and_energy_on_particles,
        _total_energies,
        _total_kinetic_energy,
        _total_potential_energy,
        _gravitational_force_and_energy_on_particles,
    )

from scipy.spatial import KDTree

def compute_LJ_force(xi, xj, LJ_R2, LJ_energy):
    rv = xj - xi
    rSq = rv.dot(rv)
    r2 = LJ_R2 / rSq
    r6 = r2**3
    r12 = r6**2
    return -12*rv/rSq * LJ_energy * (r12-r6)

def compute_LJ_energy(r2, LJ_R2, LJ_energy):
    r2 = LJ_R2 / r2
    r6 = r2**3
    r12 = r6**2
    return LJ_energy * (r12-2*r6)

def compute_LJ_force_and_energy(pos, LJ_r, LJ_e, LJ_Rmax):
    LJ_R2 = LJ_r**2
    T = KDTree(pos)
    forces = np.zeros_like(pos)
    energies = np.zeros_like(pos[:,0])
    offset = compute_LJ_energy(LJ_Rmax**2, LJ_R2, LJ_e)
    for i, j in T.query_pairs(LJ_Rmax):
        F = compute_LJ_force(pos[i], pos[j], LJ_R2, LJ_e)
        r2 = ((pos[i] - pos[j])**2).sum()
        V = compute_LJ_energy(r2, LJ_R2, LJ_e) - offset
        forces[i] += F
        forces[j] -= F
        energies[i] += 0.5*V
        energies[j] += 0.5*V
    return forces, energies

def py_grav_force_and_energy_on_particles(pos, g):
    r = np.linalg.norm(pos,axis=1)
    return (
                - pos / r[:,None] * g,
                r*g,
            )



class PhysicsTest(unittest.TestCase):

    def test_total_energies(self):

        N = 1000
        K = np.random.rand(N)
        V = np.random.rand(N)
        Vij = np.random.rand(N)
        Ucpp = _total_energies(K,V,Vij)
        Unp = K+V+Vij
        assert(np.allclose(Ucpp,Unp))

    def test_total_kinetic_energy(self):

        N = 1000
        v = np.random.randn(N,2)
        Kcpp = _total_kinetic_energy(v)
        Knp = 0.5*(v**2).sum()
        assert(np.isclose(Kcpp, Knp))

    def test_total_potential_energy(self):
        N = 1000
        x = np.random.randn(N,2)
        g = np.random.rand()
        Vcpp = _total_potential_energy(x,g)
        Vnp = g*np.linalg.norm(x,axis=1).sum()
        assert(np.isclose(Vcpp, Vnp))

    def test_LJ_force_and_energy(self):
        LJ_r = 2.5
        LJ_r_Sq = LJ_r**2
        LJ_e = 3.5
        x1 = np.array([-1/np.sqrt(2)]*2) # focal particle in lower left quadrant
        x2 = np.array([+1/np.sqrt(2)]*2) # neighb. particle in upper right quad.
        r = x2-x1 # vector pointing to neighbor
        Fcpp, Vcpp =_LJ_force_and_energy(r, r.dot(r), LJ_r_Sq, LJ_e)
        Fpy = compute_LJ_force(x1, x2, LJ_r_Sq, LJ_e)
        Vpy = compute_LJ_energy(r.dot(r), LJ_r_Sq, LJ_e)

        # since |r| = 2, particles are closer than LJ_r and should repel.
        # therefore force should point to lower left quad.

        assert(np.allclose(Fcpp, Fpy))
        assert(np.isclose(Vcpp, Vpy))

    def test_LJ_force_and_energy_on_two_particles(self):
        LJ_Rmax = 4.
        LJ_r = 2.5
        LJ_r_Sq = LJ_r**2
        LJ_e = 3.5
        x1 = np.array([-1/np.sqrt(2)]*2) # focal particle in lower left quadrant
        x2 = np.array([+1/np.sqrt(2)]*2) # neighb. particle in upper right quad.
        r = x2 - x1 # vector pointing to neighbor
        Fcpp, Vcpp = _LJ_force_and_energy(r, r.dot(r), LJ_r_Sq, LJ_e)
        offset = compute_LJ_energy(LJ_Rmax**2, LJ_r**2, LJ_e)
        #Vcpp -= offset

        pos = np.array([x1,x2])
        forces_cpp, energies_cpp = _LJ_force_and_energy_on_particles(
                                        pos,
                                        LJ_r,
                                        LJ_e,
                                        LJ_Rmax,
                                    )

        assert(np.allclose(forces_cpp[0], Fcpp))
        assert(np.isclose(energies_cpp[0], (Vcpp-offset)/2))


    def test_LJ_force_and_energy_on_particles(self):
        N = 1000
        pos = np.random.randn(N,2) * 20
        LJ_Rmax = 4.
        LJ_r = 2.5
        LJ_e = 3.5
        forces_cpp, energies_cpp = _LJ_force_and_energy_on_particles(
                                        pos,
                                        LJ_r,
                                        LJ_e,
                                        LJ_Rmax,
                                    )

        forces_py, energies_py = compute_LJ_force_and_energy(
                                        pos,
                                        LJ_r,
                                        LJ_e,
                                        LJ_Rmax,
                                    )

        assert(all([np.allclose(fcpp, fpy) for fcpp, fpy in zip(forces_cpp, forces_py)]))
        assert(all([np.isclose(ecpp, epy) for ecpp, epy in zip(energies_cpp, energies_py)]))

    def test_gravitational_force_and_energy_on_particles(self):
        N = 1000
        pos = np.random.randn(N,2) * 20
        g = np.pi

        forces_cpp, energies_cpp = _gravitational_force_and_energy_on_particles(pos, g)
        forces_py, energies_py = py_grav_force_and_energy_on_particles(pos, g)

        assert(all([np.allclose(fcpp, fpy) for fcpp, fpy in zip(forces_cpp, forces_py)]))
        assert(all([np.isclose(ecpp, epy) for ecpp, epy in zip(energies_cpp, energies_py)]))

if __name__ == "__main__":

    T = PhysicsTest()
    T.test_total_energies()
    T.test_total_potential_energy()
    T.test_LJ_force_and_energy()
    T.test_LJ_force_and_energy_on_two_particles()
    T.test_LJ_force_and_energy_on_particles()
    T.test_gravitational_force_and_energy_on_particles()
