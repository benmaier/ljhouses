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
        r = x1 - x2 # vector pointing to focal particle
        Fcpp, Vcpp =_LJ_force_and_energy(r, r.dot(r), LJ_r_Sq, LJ_e)
        Fpy = compute_LJ_force(x1, x2, LJ_r_Sq, LJ_e)
        Vpy = compute_LJ_energy(r.dot(r), LJ_r_Sq, LJ_e)

        # since |r| = 2, particles are closer than LJ_r and should repel.
        # therefore force should point to lower left quad.

        assert(np.allclose(Fcpp, Fpy))
        assert(np.isclose(Vcpp, Vpy))


if __name__ == "__main__":

    T = PhysicsTest()
    T.test_total_energies()
    T.test_total_potential_energy()
    T.test_LJ_force_and_energy()
