#include <tools.h>

double norm2(const vector < double > &vec){
    double n = 0.0;
    for(auto const &x: vec){
        n += x*x;
    }
    return n;
}

double norm(const vector < double > &vec){
    return sqrt(norm2(vec));
}

double sum(const vector < double > &vec){
    double s = 0.0;
    for(auto const &x: vec){
        s += x;
    }
    return s;
}

void scale(vector < vector < double > > &vec, const double &scalar)
{
    for(auto &element: vec)
        for(auto &coord: element)
            coord *= scalar;
}

void scale(vector < double > &vec, const double &scalar)
{
    for(auto &element: vec)
        element *= scalar;
}

void set_to_zero(vector < vector < double > > &vec)
{
    for(auto &element: vec)
        for(auto &coord: element)
            coord = 0.0;
}
