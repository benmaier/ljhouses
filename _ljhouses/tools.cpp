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
