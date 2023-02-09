#ifndef PHYSICS_H
#define PHYSICS_H

#include <vector>
#include <tools.h>
#include <KDTree.h>
#include <algorithm>

using namespace std;

// This is the corresponding positions of data in neighbor-tuples that KD-Tree returns
const unsigned int NEIGHID = 2;
const unsigned int DIST2 = 1;
const unsigned int DIFFVEC = 0;

double total_kinetic_energy(
                        const vector < vector < double > > velocities
                     );

void update_kinetic_energy(
                        const vector < vector < double > > velocities,
                        vector <double> &kinetic_energies
                     );

void update_total_energies(
                        const vector <double> &kinetic_energies,
                        const vector <double> &gravitational_energies,
                        const vector <double> &LJ_energies,
                        vector <double> &total_energies
                   );

double total_potential_energy(
            const vector < vector < double > > positions,
            const double &g
        );

pair <
        vector <double>,
        double
> LJ_force_and_energy(
                const vector <double> &r_pointing_towards_query_position,
                const double &rSquared,
                const double &LJ_r,
                const double &LJ_e
             );

void update_LJ_force_and_energy_on_particles(
            const vector < vector <double> > &positions,
            const double &LJ_r,
            const double &LJ_e,
            const double &LJ_Rmax,
            vector <vector <double> > &forces,
            vector <double> &LJ_energies
        );

void update_gravitational_force_and_energy_on_particles(
            const vector < vector < double > > positions,
            const double &g,
            vector <vector <double> > &forces,
            vector <double> &gravitational_energies
        );

#endif
