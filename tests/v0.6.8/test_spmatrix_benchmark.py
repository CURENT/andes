import unittest

import cvxopt
import numpy as np
import time

from cvxopt import matrix, spmatrix  # NOQA


class TestCVXOPTBenchmark(unittest.TestCase):
    def setUp(self):
        # configurable stuff
        self.shape = 1000
        self.sparsity = 0.01
        self.n_loops = 1
        # computed
        self.n_element = int(self.shape ** 2 * self.sparsity)
        self.i1 = matrix(np.random.randint(0, self.shape, self.n_element, dtype=int))
        self.j1 = matrix(np.random.randint(0, self.shape, self.n_element, dtype=int))
        self.v = matrix(np.ones(self.n_element), (self.n_element, 1), 'd')
        self.v0 = matrix(np.zeros(self.n_element), (self.n_element, 1), 'd')
        # self.i1 = np.arange(0, self.shape)
        # self.j1 = np.arange(0, self.shape)
        # self.v = np.ones(self.shape)

        self.spmat0 = spmatrix(self.v0, self.i1, self.j1, (self.shape, self.shape), 'd')
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

    def test_spmatrix_inplace_ipadd(self):
        try:
            t0 = time.time()
            id0 = id(self.spmat1)

            for i in range(self.n_loops):
                self.spmat1.ipadd(self.v, self.i1, self.j1)
            t_elapsed = (time.time() - t0) / self.n_loops

            print("spmatrix 1 ipadd took {} s".format(t_elapsed))
            if id(self.spmat1) == id0:
                print("  spmat1 address did not change")
            else:
                print("  spmat1 address changed!!")

            t0 = time.time()
            id0 = id(self.spmat2)
            for i in range(self.n_loops):
                self.spmat2.ipadd(self.v, self.j1, self.i1)
            t_elapsed = (time.time() - t0) / self.n_loops
            print("spmatrix 2 ipadd {} s".format(t_elapsed))
            if id(self.spmat2) == id0:
                print("  spmat2 address did not change")
            else:
                print("  spmat2 address changed!!")
        except AttributeError:
            pass

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

    def test_spmatrix_inplace_add_to_zeros(self):
        t0 = time.time()
        id0 = id(self.spmat0)
        for i in range(int(self.n_loops/2)):
            self.spmat0 += self.spmat1
            self.spmat0 -= self.spmat1
        t_elapsed = (time.time() - t0) / self.n_loops / 2
        print("Inplace add of spmat1 to spmat0 took {} s".format(t_elapsed))

        if id(self.spmat0) == id0:
            print("  spmat0 address did not change")
        else:
            print("  spmat0 address changed!!")

    def test_spmatrix_inplace_add_partial(self):
        try:
            self.spmat += self.spmat2

            t0 = time.time()
            for i in range(self.n_loops):
                self.spmat += self.spmat1
            t_elapsed = (time.time() - t0) / self.n_loops
            print("Inplace add of spmat1 to spmat took {} s".format(t_elapsed))

            t0 = time.time()
            for i in range(self.n_loops):
                self.spmat.ipadd(self.v, self.i1, self.j1)
            t_elapsed = (time.time() - t0) / self.n_loops
            print("ipadd of spmat1 to spmat took {} s".format(t_elapsed))
        except AttributeError:
            pass

    def test_spmat1_multiply_identity(self):
        t0 = time.time()
        id0 = id(self.spmat1)
        identity = cvxopt.spdiag(matrix(1, (self.shape, 1), 'd'))

        for _ in range(self.n_loops):
            self.spmat1 = self.spmat1 * identity

        t_elapsed = (time.time() - t0) / self.n_loops
        print("spmat switch_times identity matrix took {} s".format(t_elapsed))

        if id(self.spmat1) == id0:
            print("  spmat1 address did not change")
        else:
            print("  spmat1 address changed!!")

    def test_incremental_spmatrix_creation(self):
        shape = 5000
        n_batch = 200
        n_size = 20  # number of elements each batch
        n_loop = 50
        Il = []
        Jl = []
        Vl = []
        for i in range(n_batch):
            Il.append(matrix(np.random.randint(0, shape, n_size)))
            Jl.append(matrix(np.random.randint(0, shape, n_size)))
            Vl.append(matrix(np.ones((n_size, 1))))

        Ss_shape = spmatrix(0,
                            np.ravel(matrix(Il)),
                            np.ravel(matrix(Jl)),
                            (shape, shape), 'd')

        t0 = time.time()
        for nl in range(n_loop):
            Im = matrix(Il)
            Jm = matrix(Jl)
            Vm = matrix(Vl)
            for ii, jj, vv in zip(Im, Jm, Vm):
                Ss_shape[ii, jj] += vv

        print("Time for incrementally build with for loop took {} s".format((time.time() - t0) / n_loop))

        t0 = time.time()
        for nl in range(n_loop):
            Im = matrix(Il)
            Jm = matrix(Jl)
            Vm = matrix(Vl)
            Ss = spmatrix(Vm, Im, Jm, (shape, shape), 'd')
        print("Time for incremental triplet building took {} s".format((time.time() - t0) / n_loop))

        t0 = time.time()
        for nl in range(n_loop):
            Im = matrix([])
            Jm = matrix([])
            for i in range(n_batch):
                Im = matrix([Im, Il[i]])
                Jm = matrix([Jm, Jl[i]])
            Ss = spmatrix(0, Im, Jm, (shape, shape), 'd')

            for i in range(n_batch):
                Ss += spmatrix(Vl[i], Il[i], Jl[i], (shape, shape), 'd')

        print("Time for incremental spmatrix building took {} s".format((time.time() - t0) / n_loop))

        try:
            t0 = time.time()
            for nl in range(n_loop):
                Im = matrix([])
                Jm = matrix([])
                for i in range(n_batch):
                    Im = matrix([Im, Il[i]])
                    Jm = matrix([Jm, Jl[i]])
                Ss = spmatrix(0, Im, Jm, (shape, shape), 'd')

                for i in range(n_batch):
                    Ss.ipadd(Vl[i], Il[i], Jl[i])

            print("Time for ipadd spmatrix building took {} s".format((time.time() - t0) / n_loop))

        except AttributeError:
            pass
