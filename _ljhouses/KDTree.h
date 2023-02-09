#pragma once

//
//  KDTree.h
//  VerletCollision
//
//  Created by Ben F. Maier on 08.07.22.
//

#ifndef KDTree_h
#define KDTree_h

/*
 * file: KDTree.hpp
 * author: J. Frederico Carvalho
 *
 * This is an adaptation of the KD-tree implementation in rosetta code
 *  https://rosettacode.org/wiki/K-d_tree
 * It is a reimplementation of the C code using C++.
 * It also includes a few more queries than the original
 *
 */

#include <tuple>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

using namespace std;

using point_t = vector< double >;
using indexArr = vector< size_t >;
using pointIndex = typename std::pair< vector< double >, size_t >;

class KDNode {
   public:
    using KDNodePtr = shared_ptr< KDNode >;
    size_t index;
    point_t x;
    KDNodePtr left;
    KDNodePtr right;

    // initializer
    KDNode();
    KDNode(const point_t &, const size_t &, const KDNodePtr &,
           const KDNodePtr &);
    KDNode(const pointIndex &, const KDNodePtr &, const KDNodePtr &);
    ~KDNode();

    // getter
    double coord(const size_t &);

    // conversions
    explicit operator bool();
    explicit operator point_t();
    explicit operator size_t();
    explicit operator pointIndex();
};

using KDNodePtr = shared_ptr< KDNode >;

KDNodePtr NewKDNodePtr();

// square euclidean distance
inline double dist2(const point_t &, const point_t &);
inline pair < double, point_t >
    dist2_and_diff(const point_t &, const point_t &);
inline double dist2(const KDNodePtr &, const KDNodePtr &);

// euclidean distance
inline double dist(const point_t &, const point_t &);
inline double dist(const KDNodePtr &, const KDNodePtr &);

// Need for sorting
class comparer {
   public:
    size_t idx;
    explicit comparer(size_t idx_);
    inline bool compare_idx(
        const pair< vector< double >, size_t > &,  //
        const pair< vector< double >, size_t > &   //
    );
};

using pointIndexArr = typename std::vector< pointIndex >;

inline void sort_on_idx(const pointIndexArr::iterator &,  //
                        const pointIndexArr::iterator &,  //
                        size_t idx);

using pointVec = vector< point_t >;

class KDTree {
    KDNodePtr root;
    KDNodePtr leaf;

    KDNodePtr make_tree(const pointIndexArr::iterator &begin,  //
                        const pointIndexArr::iterator &end,    //
                        const size_t &length,                  //
                        const size_t &level                    //
    );

   public:
    KDTree() = default;
    explicit KDTree(pointVec point_array);

   private:
    KDNodePtr nearest_(           //
        const KDNodePtr &branch,  //
        const point_t &pt,        //
        const size_t &level,      //
        const KDNodePtr &best,    //
        const double &best_dist   //
    );

    // default caller
    KDNodePtr nearest_(const point_t &pt);

   public:
    point_t nearest_point(const point_t &pt);
    size_t nearest_index(const point_t &pt);
    pointIndex nearest_pointIndex(const point_t &pt);

   private:
    vector <tuple< vector< double >, double, size_t > >  neighborhood_(  //
        const KDNodePtr &branch,  //
        const point_t &pt,        //
        const double &rad,        //
        const size_t &level       //
    );

   public:
    vector <tuple< vector< double >, double, size_t > >  neighborhood(  //
        const point_t &pt,       //
        const double &rad);

    pointVec neighborhood_points(  //
        const point_t &pt,         //
        const double &rad);

    indexArr neighborhood_indices(  //
        const point_t &pt,          //
        const double &rad);
};

#endif /* KDTree_h */
