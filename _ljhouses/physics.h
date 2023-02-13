#ifndef PHYSICS_H
#define PHYSICS_H

#include <vector>
#include <tools.h>
#include <KDTree.h>
#include <algorithm>
#include <random>
#include <math.h>

using namespace std;

// This is the corresponding positions of data in neighbor-tuples that KD-Tree returns
const unsigned int DIFFVEC = 0;
const unsigned int DIST2 = 1;
const unsigned int NEIGHID = 2;

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

double total_interaction_energy(
            const vector < vector <double> > &positions,
            const double &LJ_r,
            const double &LJ_e,
            const double &LJ_Rmax
        );

double total_potential_energy(
            const vector < vector < double > > positions,
            const double &g
        );

void LJ_force_and_energy(
                const vector <double> &r_pointing_towards_neighbor,
                const double &rSquared,
                const double &LJ_r,
                const double &LJ_e,
                vector <double> &force,
                double &energy
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

class StochasticBerendsenThermostat {

    private:

        bool is_active;
        mt19937 rnd_gen;
        normal_distribution<double> randn{0.0,1.0};
        double velocity_scale_lower_bound;
        double velocity_scale_upper_bound;
        double target_kinetic_energy;
        double dt_over_tau;
        double diffusion_scale;

    public:

        StochasticBerendsenThermostat(){
            is_active = false;
            /*
            target_kinetic_energy = -1;
            dt_over_tau = -1;
            diffusion_scale = -1;
            */

        }

        StochasticBerendsenThermostat(
                    const size_t &N,
                    const size_t &dim,
                    const double &target_root_mean_squared_velocity,
                    const double &berendsen_tau_as_multiple_of_dt = 10.0,
                    const double &_velocity_scale_lower_bound = 0.9,
                    const double &_velocity_scale_upper_bound = 1.1,
                    const size_t &seed = 0
                )
        {
            is_active = true;

            velocity_scale_lower_bound = _velocity_scale_lower_bound;
            velocity_scale_upper_bound = _velocity_scale_upper_bound;

            const double Nf = dim*N;
            diffusion_scale = 2*sqrt(target_kinetic_energy*dt_over_tau/Nf);
            dt_over_tau = 1/berendsen_tau_as_multiple_of_dt;
            target_kinetic_energy = 0.5 * N * pow(target_root_mean_squared_velocity,2);

            if (seed > 0)
                rnd_gen.seed(seed);
        }

        void thermalize(
            vector < vector < double > > velocities,
            const double &current_kinetic_energy
        );

};

#endif
