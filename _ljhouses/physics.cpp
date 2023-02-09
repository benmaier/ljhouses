#include <physics.h>

double total_kinetic_energy(
                        const vector < vector < double > > velocities
                     )
{

    double K = 0.0;
    for(auto const &v: velocities){
        K += norm2(v);
    }
    return 0.5*K;
    
}

void update_total_energies(
                        const vector <double> &kinetic_energies,
                        const vector <double> &gravitational_energies,
                        const vector <double> &LJ_energies,
                        vector <double> &total_energies
                   )
{
    auto K = kinetic_energies.begin();
    auto V = gravitational_energies.begin();
    auto Vij = LJ_energies.begin();
    for(auto &H: total_energies)
    {
        H = (*K) + (*V) + (*Vij);

        K++;
        V++;
        Vij++;
    }
}

void update_kinetic_energy(
                        const vector < vector < double > > velocities,
                        vector <double> &kinetic_energies
                     )
{
    auto this_energy = kinetic_energies.begin();
    for(auto const &v: velocities){
        (*this_energy) = 0.5*norm2(v);
        this_energy++;
    }
}


double total_potential_energy(
            const vector < vector < double > > positions,
            const double &g
        )
{
    double V = 0.0;
    for(auto const &x: positions){
        V += norm(x)*g;
    }
    return V;
}

pair <
        vector <double>,
        double
> LJ_force_and_energy(            
                const vector <double> &r_pointing_towards_query_position,
                const double &rSquared,
                const double &LJ_r_Squared,
                const double &LJ_e
             )
{

    vector <double> force;
    double energy;
    double r2 = LJ_r_Squared/rSquared;
    double r6 = r2*r2*r2;
    double r12 = r6*r6;

    double energy_base = LJ_e * (r12 - r6);
    energy = energy_base - LJ_e * r6; //convoluted way of saying LJ_e * (r12 - 2*r6)       

    for(auto const &x_i: r_pointing_towards_query_position){
        force.push_back(
                    // the minus sign is already contained in how the
                    // KD-Tree computes the diff vector r
                    (x_i/rSquared) * 12 * energy_base
                );
    }

    return make_pair(force, energy);
}

void update_LJ_force_and_energy_on_particles(
            const vector < vector <double> > &positions,
            const double &LJ_r,
            const double &LJ_e,
            const double &LJ_Rmax,
            vector < vector < double > > &forces,
            vector < double > &LJ_energies
        )
{

    double const LJ_r_Squared = LJ_r*LJ_r;
    double const Rmax2 = LJ_r_Squared / (LJ_Rmax*LJ_Rmax);
    double const LJ_offset = LJ_e * (pow(Rmax2,6)-pow(Rmax2,3));

    for(auto &LJ_ener: LJ_energies)
        LJ_ener = 0.0;

    KDTree T(positions);

    size_t N = positions.size();
    vector < vector < size_t > > neighbors_for_which_pair_force_has_already_been_added(N);

    size_t this_particle_id = 0;
    auto this_force = forces.begin();
    auto this_energy = LJ_energies.begin();

    for(auto const &pos: positions)
    {

        vector < 
            tuple < 
                vector <double>,
                double,
                size_t
            >
        > neighbors = T.neighborhood(pos, LJ_Rmax);

        for(auto const &neigh_entry: neighbors)
        {
            size_t neigh = get<NEIGHID>(neigh_entry);

            vector < size_t > neighs_for_which_pair_has_been_added = 
                neighbors_for_which_pair_force_has_already_been_added[this_particle_id];
            const bool pair_force_has_already_been_added = 
                find(
                        neighs_for_which_pair_has_been_added.begin(),
                        neighs_for_which_pair_has_been_added.end(),
                        neigh
                   ) != neighs_for_which_pair_has_been_added.end();

            if (!pair_force_has_already_been_added)
            {

                pair <
                        vector <double>,
                        double
                > force_and_energy = 
                    LJ_force_and_energy(
                           get<DIFFVEC>(neigh_entry),
                           get<DIST2>(neigh_entry),
                           LJ_r_Squared,
                           LJ_e
                        );
                
                auto this_force_coord = this_force->begin();
                auto neighbor_force_coord = forces[neigh].begin();
                for(auto const &F_i: force_and_energy.first){
                    (*this_force_coord) += F_i;
                    (*neighbor_force_coord) -= F_i;
                    this_force_coord++;
                    neighbor_force_coord++;
                }

                double this_pairwise_energy = 0.5*(force_and_energy.second - LJ_offset);
                (*this_energy) += this_pairwise_energy;
                LJ_energies[neigh] += this_pairwise_energy;

                neighbors_for_which_pair_force_has_already_been_added[neigh].push_back(this_particle_id);

            }
        }

        this_particle_id++;
        this_force++;
        this_energy++;
    }

}

void update_gravitational_force_and_energy_on_particles(
            const vector < vector < double > > positions,
            const double &g,
            vector < vector < double > > &forces,
            vector < double > &gravitational_energies
        )
{

    auto this_force = forces.begin();
    auto this_grav_energy = gravitational_energies.begin();

    for(auto const &pos: positions)
    {
        double r = norm(pos);
        auto this_force_coord = this_force->begin();

        for(auto const &x_i: pos)
        {
            (*this_force_coord) -= g*x_i/r;
            this_force_coord++;
        }

        (*this_grav_energy) = g*r;

        this_force++;
        this_grav_energy++;
    }
}

