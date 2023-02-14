/*
 * The MIT License (MIT)
 * Copyright (c) 2023, Benjamin F. Maier
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall
 * be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-
 * INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
 * OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <KDTree.h>
#include <tools.h>
#include <physics.h>
#include <simulations.h>
#include <python_api_tools.h>

using namespace pybind11::literals; // for using "parametername"_a instead of py::arg("parametername")
using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_ljhouses, m)
{
    m.doc() = R"pbdoc(
        C++-core of ljhouses.

        .. currentmodule:: _ljhouses

        Classes
        -------

        .. autosummary::
            :toctree: _generate
            simulation

    )pbdoc";

    
    py::class_<KDTree>(m, "_KDTree", R"pbdoc(A k-d-Tree. Actually just needed internally. This here is only a miniscule interface to Python for testing purposes)pbdoc")
        .def(py::init<>(),"Initialize an empty tree.")
        .def(py::init< vector < vector < double > > &
                     >(),
             py::arg("positions"/*, "List of 2-Tuples containing (x, y)-positions"*/),
             "Initialize a tree given a list of positions.")
        .def("query_ball",
             &KDTree::neighborhood,
             R"pbdoc(Return a list of tuples, each tuple contains (i) difference vector to from match to query point, (ii) squared distance of match to query point, and (iii) the index of the match. Ignores points at zero distance.)pbdoc"
             );


    m.def("_total_energies",
          &update_total_energies_PYTHON,
          R"pbdoc(For testing purposes. Each element of the last parameter will be updated with the sum of the corresponding elements in the first three parameters)pbdoc",
          py::arg("kinetic_energies"),
          py::arg("gravitational_energies"),
          py::arg("LJ_energies")
         );

    m.def("_norm2", &norm2, R"pbdoc(Squared 2-norm of a vector.)pbdoc");
    m.def("_norm", &norm, R"pbdoc(2-norm of a vector.)pbdoc");
    m.def("_sum", &sum, R"pbdoc(Sum of all elements of a vector.)pbdoc");

    m.def("_total_kinetic_energy",
           &total_kinetic_energy,
           R"pbdoc(Compute total kinetic energy given a list of velocities.)pbdoc",
           "velocities"_a
         );

    m.def("_total_potential_energy",
           &total_potential_energy,
           R"pbdoc(Compute total potential energy given a list of positions.)pbdoc",
           "positions"_a,
           "gravitational_constant"_a
         );

    m.def("_total_interaction_energy",
           &total_interaction_energy,
           R"pbdoc(Compute total LJ interaction energy given a list of positions.)pbdoc",
           "positions"_a,
           "LJ_r"_a,
           "LJ_e"_a,
           "LJ_Rmax"_a
         );

    m.def("_LJ_force_and_energy",
           &LJ_force_and_energy_PYTHON,
           R"pbdoc(Compute the LJ-force and -energy between a single pair of particles.)pbdoc",
           "r_pointing_towards_neighbor"_a,
           "rSquared"_a,
           "LJ_r_Squared"_a,
           "LJ_e"_a
         );

    m.def("_LJ_force_and_energy_on_particles",
           &update_LJ_force_and_energy_on_particles_PYTHON,
           R"pbdoc(Compute the LJ-forces and -energies between pairs of particles that lie within a certain distance.)pbdoc",
           "positions"_a,
           "LJ_r"_a,
           "LJ_e"_a,
           "LJ_Rmax"_a
          );

    m.def("_gravitational_force_and_energy_on_particles",
           &update_gravitational_force_and_energy_on_particles_PYTHON,
           R"pbdoc(Compute the gravitational forces and energies for a list of positions.)pbdoc",
           "positions"_a,
           "gravitational_constant"_a
          );


    py::class_<StochasticBerendsenThermostat>(m,  "StochasticBerendsenThermostat", R"pbdoc(A stochastic Berendsen thermostat to rescale velocities in a way that replicates the canonical ensemble in the long term.)pbdoc")
        .def(py::init<>(),"Initializing without argument will give you an inactive thermostat leading to a microcanonical (NVE) ensemble.")
        .def(py::init< 
                       const double &,
                       const size_t &,
                       const size_t &,
                       const double &,
                       const double &,
                       const double &,
                       const size_t &
                     >(),
             py::arg("target_root_mean_squared_velocity"),
             py::arg("N"),
             py::arg("dim") = 2,
             py::arg("berendsen_tau_as_multiple_of_dt") = 10.0,
             py::arg("velocity_scale_lower_bound") = 0.9,
             py::arg("velocity_scale_upper_bound") = 1.1,
             py::arg("seed") = 0,            
             "Initialize stochastic Berendsen thermostat."
            )
        .def("get_thermalized_velocities",
             &StochasticBerendsenThermostat::get_thermalized_velocities,
             "Returns rescaled velocities according to the rules of the stochastic Berendsen thermostat.",
             py::arg("velocities"),
             py::arg("current_kinetic_energy") = -1.0
            )
        .def_readwrite("is_active", &StochasticBerendsenThermostat::is_active)
        ;

    m.def("simulate_once", &simulate_once_PYTHON, "Run a simple single simulation that returns the final configuration as well as the current energies.",
        "positions"_a,
        "velocities"_a,
        "accelerations"_a,
        "dt"_a,
        "LJ_r"_a,
        "LJ_e"_a,
        "LJ_Rmax"_a,
        "g"_a,
        "Nsteps"_a,
        "thermostat"_a
    );

    m.def("simulation", &simulation, R"pbdoc(Run a chain of consecutive simulations, returning a list of configuration samples as well as time series for the respective energies in the system.)pbdoc",
        "dt"_a,
        "N_sampling_rounds"_a,
        "N_steps_per_sample"_a,
        "max_samples"_a,
        "LJ_r"_a,
        "LJ_e"_a,
        "LJ_Rmax"_a,
        "g"_a,
        "positions"_a,
        "velocities"_a,
        "accelerations"_a,
        "thermostat"_a
    );

}
