import unittest

import numpy as np

from ljhouses import (
        _LJ_force_and_energy,
        _LJ_force_and_energy_on_particles,
        _total_energies,
        _total_kinetic_energy,
        _total_potential_energy,
        _gravitational_force_and_energy_on_particles,
        StochasticBerendsenThermostat,
        NVEThermostat,
        simulate_once,
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

def update_verlet(x, v, a, dt, LJ_r, LJ_e, LJ_Rmax, g):

    x += v * dt + a * 0.5*dt*dt

    anew, pot_energy = py_grav_force_and_energy_on_particles(x, g)

    LJ_force, LJ_energy = compute_LJ_force_and_energy(x, LJ_r, LJ_e, LJ_Rmax)
    anew += LJ_force

    v += (anew + a) * 0.5 * dt

    a = anew

    return (x, v, a, _total_kinetic_energy(v), pot_energy.sum(), LJ_energy.sum())



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

    def test_simulate_once(self):

        thermostat = NVEThermostat()

        N = 100
        np.random.seed(2)
        x = np.random.rand(N,2)*40
        v = np.random.randn(N,2)
        a = np.random.randn(N,2)
        dt = 0.01
        LJ_r = 2.5
        LJ_e = 3.5
        LJ_Rmax = 4*LJ_e
        g = 0.1
        Nsteps = 1
        result_cpp = simulate_once(
                x,v,a,
                dt,
                LJ_r,
                LJ_e,
                LJ_Rmax,
                g,
                Nsteps,
                thermostat,
            )
        vec_cpp = xcpp, vcpp, acpp = result_cpp[:3]
        enr_cpp = Kcpp, Vcpp, Vijcpp = result_cpp[3:]

        result_py = update_verlet(
                x,v,a,
                dt,
                LJ_r,
                LJ_e,
                LJ_Rmax,
                g,
            )

        vec_py = xpy, vpy, apy = result_py[:3]
        enr_py = Kpy, Vpy, Vijpy = result_py[3:]

        for vpy, vcpp in zip(vec_py, vec_cpp):
            assert(all([np.allclose(fcpp, fpy) for fcpp, fpy in zip(vcpp, vpy)]))

        assert(np.allclose(enr_py, enr_cpp))


    def test_berendsen_thermostat(self):

        N = 1000

        init_root_v2 = 1.0
        trg_root_v2 = 10.0
        Ktrg = 0.5*N*trg_root_v2**2

        #kinetic gas theory (https://en.wikipedia.org/wiki/Kinetic_theory_of_gases)
        # K/Nf = kB T/2; natural units kB = 1 and Nf = 2*N in 2D
        Ttrg = Ktrg / N
        beta = 1/Ttrg

        v = np.random.randn(N,2) * init_root_v2
        therm = StochasticBerendsenThermostat(trg_root_v2,N,velocity_scale_upper_bound=1.9,velocity_scale_lower_bound=0.1)

        nsteps = 20_000
        K = _total_kinetic_energy(v)
        Ks = [K]
        for i in range(nsteps):
            v = therm.get_thermalized_velocities(v, K)
            K = _total_kinetic_energy(v)
            Ks.append(K)

        #import matplotlib.pyplot as pl

        # in equilibrium, K should 
        # follow an Erlang distribution (https://en.wikipedia.org/wiki/Erlang_distribution)
        # as per Eq. (3) of https://arxiv.org/abs/0803.4060v1
        # with k = N and lambda = beta.
        # This distribution has mean k/lambda and variance k/lambda^2
        K = np.array(Ks[1000:])

        #print(f"{K.mean()=}", f"expected {N/beta=}")
        #print(f"{K.std()=}", f"expected {np.sqrt(N)/beta=}")

        assert(np.isclose(K.mean(), N/beta, rtol=1e-2))
        assert(np.isclose(K.std(), np.sqrt(N)/beta, rtol=5e-2))


        # comment this out if you want to to see a histogram
        #from scipy.stats import erlang

        #pdf, be, _ = pl.hist(K, bins=200, density=True)
        #x = 0.5*(be[1:] + be[:-1])
        #pl.plot(x,erlang.pdf(x, N, scale=1/beta))

        #pl.show()



if __name__ == "__main__":

    T = PhysicsTest()
    T.test_total_energies()
    T.test_total_potential_energy()
    T.test_LJ_force_and_energy()
    T.test_LJ_force_and_energy_on_two_particles()
    T.test_LJ_force_and_energy_on_particles()
    T.test_gravitational_force_and_energy_on_particles()
    T.test_simulate_once()
    #T.test_berendsen_thermostat()
