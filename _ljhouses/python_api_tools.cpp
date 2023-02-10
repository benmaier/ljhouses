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

