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

    
    py::class_<KDTree>(m, "KDTree", R"pbdoc(A k-d-Tree. Only miniscule interface to Python for testing purposes)pbdoc")
        .def(py::init<>(),"Initialize an empty tree.")
        .def(py::init< vector < vector < double > > &
                     >(),
             py::arg("positions"/*, "List of 2-Tuples containing (x, y)-positions"*/),
             "Initialize a tree given a list of positions.")
        .def("query_ball",
             &KDTree::neighborhood,
             R"pbdoc(Return a list of tuples, each tuple cotains (i) difference vector to from match to query point, (ii) squared distance of match to query point, and (iii) the index of the match. Ignores points at zero distance.)pbdoc"
             );

}
