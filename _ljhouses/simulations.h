#ifndef __SIMULATIONS_H__
#define __SIMULATIONS_H__

#include <physics.h>
#include <tools.h>

using namespace std;

void update_verlet(
        vector < vector < double > > &positions,
        vector < vector < double > > &velocities,
        vector < vector < double > > &accelerations,
        const double &dt,
        const double &LJ_r,
        const double &LJ_e,
        const double &LJ_Rmax,
        const double &g,
        vector < vector < double > > &add_forces,
        vector < double > &get_potential_energies,
        vector < double > &get_LJ_energies
    );

void simulate_once(
        vector < vector < double > > &positions,
        vector < vector < double > > &velocities,
        vector < vector < double > > &accelerations,
        const double &dt,
        const double &LJ_r,
        const double &LJ_e,
        const double &LJ_Rmax,
        const double &g,
        const size_t &Nsteps,
        StochasticBerendsenThermostat &thermostat,
        vector < double > &time,
        vector < double > &kinetic_energy,
        vector < double > &potential_energy,
        vector < double > &interaction_energy
    );

tuple <
    vector <  // vector containing samples
        tuple <
            vector < vector < double > >,
            vector < vector < double > >,
            vector < vector < double > >
        >
    >,
    vector < double >, // vector containing times of sampling energy
    vector < double >, // kinetic energy
    vector < double >, // potential energy
    vector < double >  // interaction energy
> simulation(
        const double &dt,
        const size_t &N_sampling_rounds,
        const size_t &N_steps_per_sample,
        const size_t &max_samples,
        const double &LJ_r,
        const double &LJ_e,
        const double &LJ_Rmax,
        const double &g,
        vector < vector < double > > &positions,
        vector < vector < double > > &velocities,
        vector < vector < double > > &accelerations,
        StochasticBerendsenThermostat &thermostat
    );

#endif
