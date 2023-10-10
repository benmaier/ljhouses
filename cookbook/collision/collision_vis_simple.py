from ljhouses import StochasticBerendsenThermostat, NVEThermostat
from ljhouses.vis import visualize_collisions
from ljhouses.pythonsims import simulate_collisions_once
from fincoretails import unipareto, powpareto
from ljhouses.tools import get_ideal_gas_from_theory
import numpy as np

if __name__=="__main__":     # pragma: no cover


    alpha = 4
    xmin = 1
    L = 1

    N = 2_000
    radiuses = powpareto.sample(N, alpha, xmin)
    radiuses = L/np.sqrt(N)/np.pi * radiuses

    x = get_ideal_gas_from_theory(N, L/2)
    #v = np.zeros_like(v)
    simulation_kwargs = dict(
            positions = x,
            radiuses = radiuses,
            collision_strength=1.0,
        )


    visualize_collisions(simulation_kwargs,
              config={'n_circle_segments':40},
              )
