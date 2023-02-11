import unittest

import numpy as np

from ljhouses import (
        _norm,
        _norm2,
        _sum,
    )


class ToolsTest(unittest.TestCase):

    def test_norm(self):
        N = 1000
        a = np.random.randn(N,)
        assert(np.isclose(_norm(a), np.linalg.norm(a)))

    def test_norm2(self):
        N = 1000
        a = np.random.randn(N,)
        assert(np.isclose(_norm2(a), (a**2).sum()))

    def test_sum(self):
        N = 1000
        a = np.random.randn(N,)
        assert(np.isclose(_sum(a), a.sum()))



if __name__ == "__main__":

    T = ToolsTest()
    T.test_norm()
    T.test_norm2()
    T.test_sum()
