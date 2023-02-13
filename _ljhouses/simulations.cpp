#include <simulations.h>

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
    )
{
    const double dtHalf = 0.5*dt;
    const double dtHalfSq = 0.5*dt*dt;

    auto v = velocities.begin();
    auto a = accelerations.begin();
    for(vector <double> &x: positions)
    {
        auto _v = v->begin();
        auto _a = a->begin();
        for(double &_x: x)
        {
            _x += (*_v) * dt + (*_a) * dtHalfSq;
            _v++;
            _a++;
        }
        v++;
        a++;
    }

    set_to_zero(add_forces);

    update_LJ_force_and_energy_on_particles(
            positions,
            LJ_r,
            LJ_e,
            LJ_Rmax,
            add_forces,
            get_LJ_energies
        );

    update_gravitational_force_and_energy_on_particles(
            positions,
            g,
            add_forces,
            get_potential_energies
        );

    v = velocities.begin();
    a = accelerations.begin();
    for(vector <double> &f: add_forces)
    {
        auto _v = v->begin();
        auto _a = a->begin();
        for(double &_f: f)
        {
            (*_v) += (_f + (*_a)) * dtHalf;
            _v++;
            _a++;
        }
        v++;
        a++;
    }

    //overwrite acc.s with new accelerations
    accelerations = add_forces;

}

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
    )
{

    const size_t N = positions.size();

    vector < vector < double > > get_forces(N, {0.0,0.0});
    vector < double > get_potential_energies(N, 0.0);
    vector < double > get_LJ_energies(N, 0.0);

    double t;
    if (time.size() > 0)
        t = time.back();
    else
        t = 0.0;

    for(size_t i=1; i<=Nsteps; ++i)
    {
        update_verlet(
                positions,
                velocities,
                accelerations,
                dt,
                LJ_r,
                LJ_e,
                LJ_Rmax,
                g,
                get_forces,
                get_potential_energies,
                get_LJ_energies
            );


        double K = total_kinetic_energy(velocities);

        // note that if the thermostat is inactive, it will leave the
        // velocities unchanged
        thermostat.thermalize(velocities, K);

        t += dt;
        time.push_back(t);
        kinetic_energy.push_back(K);
        potential_energy.push_back(sum(get_potential_energies));
        interaction_energy.push_back(sum(get_LJ_energies));
    }

}

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
    )
{
    vector <  // vector containing samples
        tuple <
            vector < vector < double > >,
            vector < vector < double > >,
            vector < vector < double > >
        >
    > samples;
    samples.push_back(make_tuple(
                positions,
                velocities,
                accelerations
             )
        );

    vector < double > time;
    vector < double > kinetic_energy;
    vector < double > potential_energy;
    vector < double > interaction_energy;

    time.push_back(0.0);
    kinetic_energy.push_back(total_kinetic_energy(velocities));
    potential_energy.push_back(total_potential_energy(positions,g));
    interaction_energy.push_back(total_interaction_energy(positions, LJ_r, LJ_r, LJ_Rmax));


    for(size_t ismpl=1; ismpl<=N_sampling_rounds; ++ismpl)
    {
        simulate_once(
            positions,
            velocities,
            accelerations,
            dt,
            LJ_r,
            LJ_e,
            LJ_Rmax,
            g,
            N_steps_per_sample,
            thermostat,
            time,
            kinetic_energy,
            potential_energy,
            interaction_energy
        );

        samples.push_back(make_tuple(
                    positions,
                    velocities,
                    accelerations
                 )
            );

        // pop front
        if (samples.size()>max_samples)
            if(!samples.empty())
                samples.erase(samples.begin());
    }

    return make_tuple(samples, time, kinetic_energy, potential_energy, interaction_energy);
}
