#ifndef PYTHON_API_TOOLS_H
#define PYTHON_API_TOOLS_H

#include <physics.h>

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

#endif
