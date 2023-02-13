#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <vector>
#include <math.h>

using namespace std;

double norm2(const vector < double > &vec);

double norm(const vector < double > &vec);

double sum(const vector < double > &vec);

void scale(vector < vector < double > > &vec, const double &scalar);

void scale(vector < double > &vec, const double &scalar);

void set_to_zero(vector < vector < double > > &vec);

#endif
