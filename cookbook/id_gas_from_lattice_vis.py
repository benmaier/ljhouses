from ljhouses import StochasticBerendsenThermostat, NVEThermostat
from ljhouses.tools import get_lattice_initial_conditions, get_ideal_gas_initial_conditions
from ljhouses.vis import visualize
import numpy as np

if __name__=="__main__":     # pragma: no cover




    N = 2000
    LJ_r = 6
    LJ_e = 0
    LJ_Rmax = 3*6
    g = 0.5
    v0 = 5.0
    dt = 0.01

    x, v, a = get_lattice_initial_conditions(N, v0, g)
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
    visualize(simulation_kwargs, N_steps_per_frame, width=800, config={'n_circle_segments':5})
