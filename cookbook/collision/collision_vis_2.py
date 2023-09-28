from ljhouses import StochasticBerendsenThermostat, NVEThermostat
from ljhouses.tools import get_lattice_initial_conditions, get_ideal_gas_initial_conditions
from ljhouses.vis import visualize
from ljhouses.pythonsims import simulate_collisions_once

if __name__=="__main__":     # pragma: no cover

    import numpy as np

    N = 1_000
    LJ_r = 8
    LJ_e = 0
    LJ_Rmax = 3*6
    g = 0.3
    v0 = 1.5
    v1 = 0.0001
    dt = 0.01

    x, v, a = get_ideal_gas_initial_conditions(N, v0, g)
    #v = np.ones_like(v) * v1
    v = np.zeros_like(v)
    print(v, a)
    thermostat = StochasticBerendsenThermostat(v1, N, berendsen_tau_as_multiple_of_dt=0.1)
    thermostat = NVEThermostat()
    simulation_kwargs = dict(
            positions = x,
            velocities = v,
            accelerations = a,
            LJ_r = LJ_r,
            LJ_e = LJ_e,
            LJ_Rmax = LJ_Rmax,
            g = g,
            dt = dt,
            thermostat = thermostat,
        )

    N_steps_per_frame = 1

    visualize(simulation_kwargs,
              N_steps_per_frame,
              width=800,
              config={'n_circle_segments':10,'update_dt':1},
              simulation_api=simulate_collisions_once,
              )
