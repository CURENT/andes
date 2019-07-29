import unittest

import cvxopt
import numpy as np
import time

from cvxopt import matrix, spmatrix, sparse  # NOQA


class TestCVXOPTBenchmark(unittest.TestCase):
    def setUp(self):
        # configurable stuff
        self.shape = 10000
        self.sparsity = 0.01
        self.n_loops = 5
        # computed
        self.n_element = int(self.shape ** 2 * self.sparsity)
        self.i1 = np.random.randint(0, self.shape, self.n_element)
        self.j1 = np.random.randint(0, self.shape, self.n_element)
        self.v = np.ones(self.n_element)
        # self.i1 = np.arange(0, self.shape)
        # self.j1 = np.arange(0, self.shape)
        # self.v = np.ones(self.shape)

        self.spmat1 = spmatrix(self.v, self.i1, self.j1, (self.shape, self.shape), 'd')
        self.spmat2 = spmatrix(self.v, self.j1, self.i1, (self.shape, self.shape), 'd')
        self.spmat = self.spmat1

    def test_build_spmatrix(self):
        t0 = time.time()

        for i in range(self.n_loops):
            self.spmat1 = spmatrix(self.v, self.i1, self.j1, (self.shape, self.shape), 'd')

        t_elapsed = (time.time() - t0) / self.n_loops

        print("Building spmatrix 1 from arrays took {} s".format(t_elapsed))

        # spmatrix 2
        t0 = time.time()
        for i in range(self.n_loops):
            self.spmat2 = spmatrix(self.v, self.j1, self.i1, (self.shape, self.shape), 'd')

        t_elapsed = (time.time() - t0) / self.n_loops

        print("Building spmatrix 2 from arrays took {} s".format(t_elapsed))

    def test_spmatrix_inplace_add(self):
        t0 = time.time()
        id0 = id(self.spmat1)

        for i in range(self.n_loops):
            self.spmat1 += self.spmat1
        t_elapsed = (time.time() - t0) / self.n_loops
        print("Inplace spmatrix 1 add to itself took {} s".format(t_elapsed))
        if id(self.spmat1) == id0:
            print("  spmat1 address did not change")
        else:
            print("  spmat1 address changed!!")

        t0 = time.time()
        id0 = id(self.spmat2)
        for i in range(self.n_loops):
            self.spmat2 += self.spmat2
        t_elapsed = (time.time() - t0) / self.n_loops
        print("Inplace spmatrix 2 add to itself took {} s".format(t_elapsed))
        if id(self.spmat2) == id0:
            print("  spmat2 address did not change")
        else:
            print("  spmat2 address changed!!")

    def test_spmatrix_noninplace_add(self):
        t0 = time.time()
        id0 = id(self.spmat)
        for i in range(self.n_loops):
            self.spmat += self.spmat2
        t_elapsed = (time.time() - t0) / self.n_loops
        print("Non-inplace add of spmatrix 1 and 2 took {} s".format(t_elapsed))
        if id(self.spmat) == id0:
            print("  spmat address did not change")
        else:
            print("  spmat address changed!!")

    def test_spmatrix_inplace_add_partial(self):
        self.spmat += self.spmat2

        t0 = time.time()
        id0 = id(self.spmat)
        for i in range(self.n_loops):
            self.spmat += self.spmat1
        t_elapsed = (time.time() - t0) / self.n_loops
        print("Inplace add of spmat1 to spmat took {} s".format(t_elapsed))

        if id(self.spmat) == id0:
            print("  spmat address did not change")
        else:
            print("  spmat address changed!!")

    def test_spmat1_multiply_identity(self):
        t0 = time.time()
        id0 = id(self.spmat1)
        identity = cvxopt.spdiag(matrix(1, (self.shape, 1), 'd'))

        for i in range(self.n_loops):
            self.spmat1 = self.spmat1 * identity

        t_elapsed = (time.time() - t0) / self.n_loops
        print("spmat times identity matrix took {} s".format(t_elapsed))

        if id(self.spmat1) == id0:
            print("  spmat1 address did not change")
        else:
            print("  spmat1 address changed!!")

    # def test_spmatrix_inplace_mod(self):
    #     t0 = time.time()
    #     for i in range(self.n_loops):
    #         self.spmat[self.i1, self.j1] = 1
    #         self.spmat[self.i2, self.j2] = 1
    #     t_elapsed = (time.time() - t0) / self.n_loops
    #     print("In-place modification of spmat took {} s".format(t_elapsed))
#
