from ljhouses import StochasticBerendsenThermostat, NVEThermostat
from ljhouses.tools import get_lattice_initial_conditions
from ljhouses.vis import visualize
import numpy as np
from ljhouses.pythonsims import get_close_to_equilibrium_initial_conditions

if __name__=="__main__":     # pragma: no cover

    N = 1_000
    LJ_r = 10
    LJ_e = 20
    LJ_Rmax = 3*LJ_r
    g = 0.15
    v0 = 5.0
    dt = 0.01

    x, v, a = get_close_to_equilibrium_initial_conditions(N, v0, LJ_r, g, dt)
    thermostat = NVEThermostat()
    thermostat = StochasticBerendsenThermostat(v0, N, berendsen_tau_as_multiple_of_dt=100)
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

    N_steps_per_frame = 10
    visualize(simulation_kwargs, N_steps_per_frame, width=800)
