/*
 * The MIT License (MIT)
 * Copyright (c) 2022, Benjamin F. Maier
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
#include <python_api_tools.h>

using namespace pybind11::literals;
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
             R"pbdoc(Return a list of tuples, each tuple cotains (i) difference vector to from match to query point, (ii) squared distance of match to query point, and (iii) the index of the match. Ignores points at zero distance.)pbdoc"
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

    m.def("_LJ_force_and_energy",
           &LJ_force_and_energy,
           R"pbdoc(Compute the LJ-force and -energy between a single pair of particles.)pbdoc",
           "r_pointing_towards_query_position"_a,
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




}
