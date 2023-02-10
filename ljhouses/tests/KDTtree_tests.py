import unittest

import numpy as np

from ljhouses import _KDTree as ljkd
from scipy.spatial import KDTree as sckd


class KDTreeTest(unittest.TestCase):

    def test_query(self):

        N = 1000
        pos = np.random.rand(N,2)
        T = ljkd(pos)
        query_point = np.array([-0.001, -0.001])
        neighs = T.query_ball(query_point, 0.1)
        indices = sorted([ n[2] for n in neighs ])

        T2 = sckd(pos)
        neighs2 = sorted(T2.query_ball_point(query_point,0.1))
        assert(set(indices) == set(neighs2))

        for diff, d2, i in neighs:
            x = pos[i]
            dr = x-query_point
            assert(np.allclose(diff, dr))
            assert(np.isclose(d2, dr.dot(dr)))

    def test_query_on_contained_point(self):
        N = 1000
        pos = np.random.rand(N,2)
        T = ljkd(pos)
        neighs = T.query_ball(pos[0], 0.1)
        indices = [ n[2] for n in neighs ]
        assert(0 not in indices)


if __name__ == "__main__":

    T = KDTreeTest()
    T.test_query()
    T.test_query_on_contained_point()
