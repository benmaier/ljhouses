#ifndef PYTHON_API_TOOLS_H
#define PYTHON_API_TOOLS_H

#include <physics.h>
#include <simulations.h>

vector <double> update_kinetic_energy_PYTHON(
                        const vector < vector < double > > velocities
                     );

vector <double> update_total_energies_PYTHON(
                        const vector <double> &kinetic_energies,
                        const vector <double> &gravitational_energies,
                        const vector <double> &LJ_energies
                   );

pair <
    vector <vector <double> >,
    vector <double>
> update_LJ_force_and_energy_on_particles_PYTHON(
            const vector < vector <double> > &positions,
            const double &LJ_r,
            const double &LJ_e,
            const double &LJ_Rmax
        );

pair <
    vector <vector <double> >,
    vector <double>
> update_gravitational_force_and_energy_on_particles_PYTHON(
            const vector < vector < double > > positions,
            const double &g
        );

pair <
    vector <double>,
    double
> LJ_force_and_energy_PYTHON(
                const vector <double> &r_pointing_towards_neighbor,
                const double &rSquared,
                const double &LJ_r,
                const double &LJ_e
             );

tuple <
    vector < vector < double > >,
    vector < vector < double > >,
    vector < vector < double > >,
    double,
    double,
    double
>
simulate_once_PYTHON(
        vector < vector < double > > &positions,
        vector < vector < double > > &velocities,
        vector < vector < double > > &accelerations,
        const double &dt,
        const double &LJ_r,
        const double &LJ_e,
        const double &LJ_Rmax,
        const double &g,
        const size_t &Nsteps,
        StochasticBerendsenThermostat &thermostat
    );

#endif
