from ljhouses import simulation, StochasticBerendsenThermostat, NVEThermostat
from ljhouses.tools import get_lattice_initial_conditions
from ljhouses.vis import visualize

if __name__=="__main__":     # pragma: no cover



    import numpy as np

    N = 1000
    LJ_r = 6
    LJ_e = 2
    LJ_Rmax = 3*6
    g = 0.5
    v0 = 5.0
    dt = 0.01

    x, v, a = get_lattice_initial_conditions(N, v0, LJ_r)
    thermostat = StochasticBerendsenThermostat(v0, N, berendsen_tau_as_multiple_of_dt=100)
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

    N_steps_per_frame = 10
    visualize(simulation_kwargs, N_steps_per_frame, width=800,height=800)
