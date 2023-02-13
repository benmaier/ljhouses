#include <python_api_tools.h>

vector <double> update_kinetic_energy_PYTHON(
                        const vector < vector < double > > velocities
                     )
{
    size_t N = velocities.size();
    vector <double> kinetic_energies(N);
    update_kinetic_energy(velocities, kinetic_energies);

    return kinetic_energies;
}

vector <double> update_total_energies_PYTHON(
                        const vector <double> &kinetic_energies,
                        const vector <double> &gravitational_energies,
                        const vector <double> &LJ_energies
                   )
{
    size_t N = kinetic_energies.size();
    vector < double > total_energies(N);
    update_total_energies(kinetic_energies,
                          gravitational_energies,
                          LJ_energies,
                          total_energies
                         );

    return total_energies;
}

pair <
    vector <vector <double> >,
    vector <double>
> update_LJ_force_and_energy_on_particles_PYTHON(
            const vector < vector <double> > &positions,
            const double &LJ_r,
            const double &LJ_e,
            const double &LJ_Rmax
        )
{
    size_t N = positions.size();
    vector <vector <double> > forces(N,{0.0,0.0});
    vector <double> energies(N);

    update_LJ_force_and_energy_on_particles(
            positions,
            LJ_r,
            LJ_e,
            LJ_Rmax,
            forces,
            energies
          );

    return make_pair(forces, energies);

}

pair <
    vector <vector <double> >,
    vector <double>
> update_gravitational_force_and_energy_on_particles_PYTHON(
            const vector < vector < double > > positions,
            const double &g
        )
{
    size_t N = positions.size();
    vector <vector <double> > forces(N,{0.0,0.0});
    vector <double> energies(N);

    update_gravitational_force_and_energy_on_particles(
                positions,
                g,
                forces,
                energies
            );

    return make_pair(forces, energies);
}

pair <
    vector <double>,
    double
> LJ_force_and_energy_PYTHON(
                const vector <double> &r_pointing_towards_neighbor,
                const double &rSquared,
                const double &LJ_r,
                const double &LJ_e
             )
{
    vector <double> force(2);
    double energy;

    LJ_force_and_energy(
            r_pointing_towards_neighbor,
            rSquared,
            LJ_r,
            LJ_e,
            force,
            energy
        );

    return make_pair(force, energy);
}

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
    )
{
    vector < double > time;
    vector < double > kinetic_energy;
    vector < double > potential_energy;
    vector < double > interaction_energy;

    simulate_once(
        positions,
        velocities,
        accelerations,
        dt,
        LJ_r,
        LJ_e,
        LJ_Rmax,
        g,
        Nsteps,
        thermostat,
        time,
        kinetic_energy,
        potential_energy,
        interaction_energy
    );

    return make_tuple(positions, velocities, accelerations,
                      kinetic_energy.back(),
                      potential_energy.back(),
                      interaction_energy.back()
                     );
}
