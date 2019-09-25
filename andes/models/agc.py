import sys
import logging
import numpy as np
import importlib
import platform

from cvxopt import mul, div, matrix, sparse, spdiag, spmatrix
from cvxopt.modeling import variable, op  # NOQA
from andes.consts import Gx, Fy0, Gy0
from andes.models.base import ModelBase
from andes.utils.math import zeros, index
from andes.utils.solver import Solver

logger = logging.getLogger(__name__)
cp = None


class BArea(ModelBase):
    """
    Balancing area class. This class defines power balancing area on top of the `Area` class for calculating
    center of inertia frequency, total inertia, expected power and area control error.
    """
    def __init__(self, system, name):
        super(BArea, self).__init__(system, name)
        self._group = 'Calculation'
        self._data.update({
            'area': None,
            'syn': None,
            'beta': 0,
        })
        self._descr.update({'area': 'Idx of Area',
                            'beta': 'Beta coefficient to multiply by the pu freq. deviation',
                            'syn': 'Indices of generators for computing COI'
                            })
        self._units.update({'syn': 'list'})
        self._mandatory.extend(['area', 'syn', 'beta'])
        self._algebs.extend(['Pexp', 'fcoi', 'ace'])
        self.calls.update({
            'gcall': True,
            'init1': True,
            'jac0': True,
        })
        self._service.extend(['P0', 'Mtot', 'M', 'usyn', 'wsyn'])
        self._fnamey.extend(['P_{exp}', 'f_{coi}', 'ace'])
        self._params.extend(['beta'])
        self._init()

    def init1(self, dae):

        for item in self._service:
            self.__dict__[item] = [[]] * self.n

        # Start with frequency
        for idx, item in enumerate(self.syn):
            self.M[idx] = self.read_data_ext('Synchronous', field='M', idx=item)
            self.Mtot[idx] = sum(self.M[idx])
            self.usyn[idx] = self.read_data_ext('Synchronous', field='u', idx=item)
            self.wsyn[idx] = self.read_data_ext('Synchronous', field='omega', idx=item)
            dae.y[self.fcoi[idx]] = sum(mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

        # Get BA Export Power
        self.copy_data_ext('Area', field='area_P0', dest='P0', idx=self.area)
        dae.y[self.Pexp] = self.P0
        dae.y[self.ace] = 0

    def gcall(self, dae):

        # the value below gets updated at each iteration in `seriesflow`
        P = self.read_data_ext('Area', field='area_P0', idx=self.area)
        dae.g[self.Pexp] = dae.y[self.Pexp] - P

        for idx, item in enumerate(self.syn):
            self.wsyn[idx] = self.read_data_ext('Synchronous', field='omega', idx=item)
            dae.g[self.fcoi[idx]] = dae.y[self.fcoi[idx]] - \
                sum(mul(self.M[idx], dae.x[self.wsyn[idx]])) / self.Mtot[idx]

        ACE = (P - self.P0) - mul(self.beta, (1 - dae.y[self.fcoi]))

        dae.g[self.ace] = dae.y[self.ace] + ACE

    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.Pexp, self.Pexp)
        dae.add_jac(Gy0, 1, self.fcoi, self.fcoi)
        dae.add_jac(Gy0, 1, self.ace, self.ace)
        dae.add_jac(Gy0, 1, self.ace, self.Pexp)
        dae.add_jac(Gy0, self.beta, self.ace, self.fcoi)


class AGCBase(ModelBase):
    """
    Base AGC class. The allocation of Pagc will be based on inverse droop (iR)
    """
    def __init__(self, system, name):
        super(AGCBase, self).__init__(system, name)
        self._group = 'AGCGroup'
        self._data.update({'BArea': None,
                           'Ki': 0.05,
                           })
        self._descr.update({'BArea': 'Idx of BArea',
                            'Ki': 'Integral gain of ACE',
                            })
        self._mandatory.extend(['BArea', 'Ki'])
        self._states.extend(['Pagc'])
        self.calls.update({'init1': True,
                           'gcall': True,
                           'fcall': True,
                           'jac0': True,
                           'gycall': True
                           })
        self._service.extend(['ace', 'iR', 'iRtot'])
        self._fnamex.extend(['P_{agc}^{total}'])
        self._params.extend(['Ki'])

    def init1(self, dae):
        self.copy_data_ext('BArea', field='ace', idx=self.BArea)

    def fcall(self, dae):
        dae.f[self.Pagc] = mul(self.Ki, dae.y[self.ace])

    def gcall(self, dae):
        pass

    def jac0(self, dae):
        dae.add_jac(Fy0, self.Ki, self.Pagc, self.ace)

    def gycall(self, dae):
        pass


class AGCSyn(AGCBase):
    """AGC for synchronous generators. This class changes the setpoints by modifying the generator pm."""
    def __init__(self, system, name):
        super(AGCSyn, self).__init__(system, name)
        self._data.update({'syn': None})
        self._descr.update({'syn': 'Indices of synchronous generators for AGC'})
        self._units.update({'syn': 'list'})
        self._mandatory.extend(['syn'])
        self._service.extend(['pm', 'usyn'])
        self._init()

    def init1(self, dae):
        super(AGCSyn, self).init1(dae)
        self.pm = [[]] * self.n
        self.iR = [[]] * self.n
        self.usyn = [[]] * self.n
        self.iRtot = [[]] * self.n

        for idx, item in enumerate(self.syn):
            self.pm[idx] = self.read_data_ext('Synchronous', field='pm', idx=item)
            self.usyn[idx] = self.read_data_ext('Synchronous', field='u', idx=item)
            self.iR[idx] = self.read_data_ext('Synchronous', field='M', idx=item)
            self.iRtot[idx] = sum(mul(self.usyn[idx], self.iR[idx]))

    def gcall(self, dae):
        super(AGCSyn, self).gcall(dae)

        # Kgen and each item in `self.pm`, `self.usyn`, and `self.Pagc` is a list
        #   Do not get rid of the `for` loop, since each of them is a matrix operation

        for idx, item in enumerate(self.syn):
            Kgen = div(self.iR[idx], self.iRtot[idx])
            dae.g[self.pm[idx]] -= mul(self.usyn[idx], Kgen, dae.x[self.Pagc[idx]])

    def gycall(self, dae):
        super(AGCSyn, self).gycall(dae)

        # Do not get rid of the for loop; for each `idx` it is a matrix operation

        for idx, item in enumerate(self.syn):
            Kgen = div(self.iR[idx], self.iRtot[idx])
            dae.add_jac(Gx, -mul(self.usyn[idx], Kgen), self.pm[idx], self.Pagc[idx])


class AGC(AGCSyn):
    """Alias for class <AGCSyn>"""
    pass


class AGCTG(AGCBase):
    """AGC class that modifies the turbine governor power reference. Links to TG1 only."""
    def __init__(self, system, name):
        super(AGCTG, self).__init__(system, name)
        self._data.update({'tg': None})
        self._mandatory.extend(['tg'])
        self._descr.update({'tg': 'Indices of turbine governors for AGC'})
        self._units.update({'tg': 'list'})
        self._service.extend(['pin', 'R', 'iR', 'iRtot'])

        self._init()

    def init1(self, dae):
        super(AGCTG, self).init1(dae)
        self.pin = [[]] * self.n
        self.R = [[]] * self.n
        self.iR = [[]] * self.n
        self.iRtot = [[]] * self.n

        for idx, item in enumerate(self.tg):
            self.pin[idx] = self.read_data_ext(model='Governor', field='pin', idx=item)
            self.R[idx] = self.read_data_ext(model='Governor', field='R', idx=item)
            self.iR[idx] = div(1, self.R[idx])
            self.iRtot[idx] = sum(self.iR[idx])

    def gcall(self, dae):
        super(AGCTG, self).gcall(dae)
        for idx, item in enumerate(self.tg):
            Ktg = div(self.iR[idx], self.iRtot[idx])
            dae.g[self.pin[idx]] += mul(Ktg, dae.x[self.Pagc[idx]])

    def gycall(self, dae):
        super(AGCTG, self).gycall(dae)
        for idx, item in enumerate(self.tg):
            Ktg = div(self.iR[idx], self.iRtot[idx])
            dae.add_jac(Gx, Ktg, self.pin[idx], self.Pagc[idx])


class AGCVSCBase(object):
    """
    Base class for AGC using VSC. Modifies the ref1 for PV or PQ-controlled VSCs. This class must be
    inherited with subclasses of AGCBase
    """
    def __init__(self, system, name):
        self.system = system
        self._data.update({'vsc': None,
                           'Rvsc': None,
                           })
        self._descr.update({'vsc': 'Indices of VSCs to control',
                            'Rvsc': 'Droop coefficients for the VSCs'})
        self._units.update({'tg': 'list',
                            'Rvsc': 'list'})
        self._mandatory.extend(['vsc', 'Rvsc'])
        self._service.extend(['uvsc', 'ref1'])
        self._init()

    def init1(self, dae):
        self.ref1 = [[]] * self.n
        self.uvsc = [[]] * self.n

        # manually convert self.Rvsc to a list of matrices
        self.Rvsc = [matrix(item) for item in self.Rvsc]
        self.iRvsc = [div(1, item) for item in self.Rvsc]

        # Only PV or PQ-controlled VSCs are acceptable
        for agc_idx, item in enumerate(self.vsc[:]):
            pv_or_pq = self.read_data_ext('VSCgroup', field="PV", idx=item) + \
                        self.read_data_ext('VSCgroup', field='PQ', idx=item)

            valid_vsc_list = list()
            valid_vsc_R = list()
            for i, (vsc_idx, valid) in enumerate(zip(item, pv_or_pq)):
                if valid:
                    valid_vsc_list.append(vsc_idx)
                    # TODO: fix the hard-coded `vsc_Idx` below
                    valid_vsc_R.append(self.Rvsc[agc_idx][i])
                else:
                    logger.warning('VSC <{}> is not a PV or PQ type, thus cannot be used for AGC.'.format(vsc_idx))
            self.vsc[agc_idx] = valid_vsc_list

        for agc_idx, item in enumerate(self.vsc):
            # skip elements that contain no valid VSC index
            if len(item) == 0:
                continue

            # retrieve status `uvsc`
            self.uvsc[agc_idx] = self.read_data_ext('VSCgroup', field='u', idx=item)
            self.ref1[agc_idx] = self.read_data_ext('VSCgroup', field='ref1', idx=item)
            # Add `Rvsc` to Mtot
            self.iRtot[agc_idx] += sum(mul(self.uvsc[agc_idx], self.iRvsc[agc_idx]))

    def gcall(self, dae):
        for agc_idx, item in enumerate(self.vsc):
            if len(item) == 0:
                continue

            Kvsc = div(self.iRvsc[agc_idx], self.iRtot[agc_idx])
            dae.g[self.ref1[agc_idx]] -= mul(self.uvsc[agc_idx], Kvsc, dae.x[self.Pagc[agc_idx]])

    def gycall(self, dae):

        for agc_idx, item in enumerate(self.vsc):
            if len(item) == 0:
                continue

            Kvsc = div(self.iRvsc[agc_idx], self.iRtot[agc_idx])
            dae.add_jac(Gx, -mul(self.uvsc[agc_idx], Kvsc), self.ref1[agc_idx], self.Pagc[agc_idx])


class AGCTGVSC(AGCTG, AGCVSCBase):
    """AGC class that modifies the turbine governor and VSC pref"""
    def __init__(self, system, name):
        AGCTG.__init__(self, system, name)
        AGCVSCBase.__init__(self, system, name)
        self._init()

    def init1(self, dae):
        AGCTG.init1(self, dae)
        AGCVSCBase.init1(self, dae)

    def jac0(self, dae):
        AGCTG.jac0(self, dae)

    def gcall(self, dae):
        AGCTG.gcall(self, dae)
        AGCVSCBase.gcall(self, dae)

    def gycall(self, dae):
        AGCTG.gycall(self, dae)
        AGCVSCBase.gycall(self, dae)

    def fcall(self, dae):
        AGCTG.fcall(self, dae)


class AGCMPC(ModelBase):
    """MPC based AGC using TG and VSC"""
    def __init__(self, system, name):
        super(AGCMPC, self).__init__(system, name)
        if platform.system() == 'Darwin':
            logger.error("** AGCMPC optimization does not work correctly on macOS!!!")

        self._group = "AGCGroup"
        self._name = "AGCMPC"
        self.param_remove('Vn')
        self.param_remove('Sn')

        self._data.update({'tg': None,
                           'avr': None,
                           'vsc': None,
                           'qw': 15000,
                           'qu': 10,
                           })
        self._params.extend(['qw', 'qu'])
        self._descr.update({'tg': 'idx for turbine governors',
                            'vsc': 'idx for VSC dynamic models',
                            'qw': 'the coeff for minimizing frequency deviation',
                            'qu': 'the coeff for minimizing input deviation'
                            })
        self._units.update({'tg': 'list', 'vsc': 'list'})
        self._mandatory.extend(['tg', 'avr'])
        self.calls.update({'init1': True,
                           'gcall': True,
                           'jac0': True,
                           'fxcall': True})

        self._service.extend(['xg10', 'pin0', 'delta0', 'omega0', 't', 'dpin0', 'x0', 'xlast'
                              'xidx', 'uidx', 'yxidx', 'sfx', 'sfu', 'sfy', 'sgx', 'sgu', 'sgy',
                              'A', 'B', 'Aa', 'Ba',
                              'obj', 'domega', 'du', 'dx', 'x', 'xpred'
                              'xa'])

        self._algebs.extend(['dpin'])
        self._fnamey.extend(r'\Delta P_{in}')
        self.solver = Solver(system.config.sparselib)
        self.H = 6
        self.uvar = None
        self.op = None
        self._linearized = False
        self._interval = 0  # AGC apply interval in seconds. 0 - continuous
        self._init()

    def init1(self, dae):
        if globals()['cp'] is None:
            try:
                globals()['cp'] = importlib.import_module('cvxpy')
            except ImportError:
                logger.error('CVXPY import error. Install optional package `cvxpy` to use AGCMPC')
                sys.exit(1)

        self.t = -1
        self.tlast = -1
        # state array x = [delta, omega, xg1]
        # input array u = [dpin]
        self.copy_data_ext('Governor', field='gen', dest='syn', idx=self.tg)
        self.copy_data_ext('Synchronous', field='delta', dest='delta', idx=self.syn)
        self.copy_data_ext('Synchronous', field='omega', dest='omega', idx=self.syn)
        self.copy_data_ext('Synchronous', field='e1d', dest='e1d', idx=self.syn)
        self.copy_data_ext('Synchronous', field='e1q', dest='e1q', idx=self.syn)
        self.copy_data_ext('Synchronous', field='e2d', dest='e2d', idx=self.syn)
        self.copy_data_ext('Synchronous', field='e2q', dest='e2q', idx=self.syn)

        self.copy_data_ext('Governor', field='xg1', dest='xg1', idx=self.tg)
        self.copy_data_ext('Governor', field='xg2', dest='xg2', idx=self.tg)
        self.copy_data_ext('Governor', field='xg3', dest='xg3', idx=self.tg)
        self.copy_data_ext('Governor', field='pin', dest='pin', idx=self.tg)

        self.copy_data_ext('AVR', field='vm', dest='vm', idx=self.avr)
        self.copy_data_ext('AVR', field='vr1', dest='vr1', idx=self.avr)
        self.copy_data_ext('AVR', field='vr2', dest='vr2', idx=self.avr)
        self.copy_data_ext('AVR', field='vfout', dest='vfout', idx=self.avr)

        dae.y[self.dpin] = 0
        self.dpin0 = zeros(self.n, 1)

        # build state/ input /other algebraic idx array
        self.xidx = matrix([self.delta, self.omega, self.e1d, self.e1q, self.e2d, self.e2q, self.xg1, self.xg2,
                            self.xg3, self.vm, self.vr1, self.vr2, self.vfout])
        self.x0 = dae.x[self.xidx]
        self.x = zeros(len(self.xidx), 1)
        self.dx = zeros(len(self.xidx), 1)
        self.xlast = dae.x[self.xidx]

        self.uidx = matrix([self.dpin])
        self.ulast = zeros(self.n, 1)
        self.dpin_calc = zeros(self.n, 1)

        self.widx = self.system.PQ.a
        self.w0 = self.system.PQ.p0
        self.wlast = matrix(self.w0)

        self.yidx = self.omega
        self.yidx_in_x = [index(self.xidx, y)[0] for y in self.yidx]
        yidx = np.delete(np.arange(dae.m), np.array(self.uidx))
        self.yxidx = matrix(yidx)

        # optimization problem
        self.uvar = cp.Variable((len(self.uidx), self.H+1), 'u')
        self.uzero = cp.Parameter((len(self.uidx), ), 'u0')
        self.xazero = cp.Parameter((2 * len(self.xidx), 1), 'xa')
        self.prob = None

        self.t_store = []
        self.xpred_store = []

    def gcall(self, dae):

        if self.t == -1:
            self.t = dae.t
            return

        if not self._linearized:
            # update the linearization points
            self._linearized = True
            self.t = dae.t
            self.tlast = dae.t
            self.sfx = dae.Fx[self.xidx, self.xidx]
            self.sfu = dae.Fy[self.xidx, self.uidx]
            self.sfy = dae.Fy[self.xidx, self.yxidx]

            self.sgx = dae.Gx[self.yxidx, self.xidx]
            self.sgu = dae.Gy[self.yxidx, self.uidx]
            self.sgw = spmatrix(1, self.widx, list(range(len(self.widx))), (len(self.yxidx), len(self.widx)))
            self.sgy = dae.Gy[self.yxidx, self.yxidx]

            # create state matrices
            self.gyigx = matrix(self.sgx)
            self.gyigu = matrix(self.sgu)
            self.gyigw = matrix(self.sgw)

            self.solver.linsolve(self.sgy, self.gyigx)
            self.solver.linsolve(self.sgy, self.gyigu)
            self.solver.linsolve(self.sgy, self.gyigw)

            self.A = (self.sfx - self.sfy * self.gyigx)
            self.B = (self.sfu - self.sfy * self.gyigu)
            self.C = -(self.sfy * self.gyigw)

            self.A = self.system.tds.h * self.A
            self.Aa = sparse([[self.A, self.A],
                              [spmatrix([], [], [], (self.A.size[0], self.A.size[1])),
                               spdiag([1] * len(self.xidx))]])
            self.Ba = sparse([self.B, self.B])
            self.Ca = sparse([self.C, self.C])

            # formulate optimization problem
            nx = len(self.xidx)
            nu = len(self.uidx)
            obj_x = 0

            xa_0 = self.xazero
            for i in range(self.H):
                # calculate Xa for each step in horizon H
                # du = cp.reshape(self.uvar[:, i+1], (nu, 1)) - self.uvar[:,i]
                du = cp.reshape(self.uvar[:, i+1] - self.uvar[:, i], (nu, 1))
                xa_i = matrix(self.Aa) * xa_0 + matrix(self.Ba) * du
                obj_x += cp.multiply(self.qw, cp.square(xa_i[nx:][self.yidx_in_x] - self.x0[self.yidx_in_x]))
                xa_0 = xa_i

            # construct the optimization problem
            self.obj_x = cp.sum(obj_x)

            self.obj_u = 0

            self.obj_u += cp.sum(
                cp.multiply(
                    np.array(self.qu).reshape((nu, )),
                    cp.sum(cp.square(self.uvar[:, 1:] - self.uvar[:, :-1]), axis=1)
                )
            )

            constraints = [self.uvar[:, 0] == self.uzero,
                           self.uvar[:, 1:] - self.uvar[:, :-1] <= 0.5,
                           self.uvar[:, 1:] - self.uvar[:, :-1] >= -0.5
                           ]

            self.prob = cp.Problem(cp.Minimize(self.obj_x + self.obj_u), constraints)

        if dae.t != self.t:
            self.t = dae.t
            nx = len(self.xidx)
            nu = len(self.uidx)

            # # update Delta x and x for current step
            self.x = dae.x[self.xidx]
            self.dx = self.x - self.xlast
            self.xa = matrix([self.dx, self.x])

            # assign values to self.uzero and self.xazero
            self.uzero.value = np.array(self.ulast).reshape((-1, ))
            self.xazero.value = np.array(self.xa).reshape((-1, 1))

            # use warm_start when possible
            if dae.t == 0:
                self.prob.solve()
            else:
                self.prob.solve(warm_start=1)

            self.dpin_calc = matrix(self.uvar.value[:, 1])

            # update every interval
            if (self.t - self.tlast) >= self._interval:
                self.tlast = self.t
                self.dpin0 = self.dpin_calc

            opt_val = self.prob.solution.opt_val
            logger.debug("t={:.4f}, obj={:.6f}, u[0]={:.6f}".format(dae.t, opt_val, self.uvar.value[0, 0]))

            self.t_store.append(self.t)
            xa_post = matrix(self.Aa) * self.xa + matrix(self.Ba) * (matrix(self.uvar.value[:, 0]) - self.ulast)
            self.xpred_store.append(xa_post[nx:][self.yidx_in_x][0])

            # # post-optimization evaluator
            # # u_val = matrix([[0, 0], [0, 0], [0, 0]])
            # u_val = matrix(self.uvar.value)
            # u_val = zeros(2, self.H)
            # obj_x = 0
            # xa_0 = self.xa
            # u_0 = self.ulast
            # for i in range(self.H):
            #     # calculate Xa for each step in horizon H
            #     du = np.reshape(u_val[:, i], (-1, 1)) - u_0
            #     xa_i = matrix(self.Aa) * xa_0 + matrix(self.Ba) * matrix(du) #+ matrix(self.Ca) * self.dw
            #     obj_x += mul(self.qw, (xa_i[nx:][self.yidx_in_x] - self.x0[self.yidx_in_x]) ** 2)
            #     xa_0 = xa_i
            #     u_0 = np.reshape(u_val[:, i], (-1, 1))
            # self.obj_x = sum(obj_x)
            # u2 = np.array(mul(u_val, u_val))
            # self.obj_u = sum(mul(self.qu, matrix(np.sum(u2, 1))))
            #
            # eval_obj = self.obj_x + self.obj_u
            # print("Post eval, t={:.4f} obj = {:.6f}, u = {:.6f}, {:.6f}".format(self.t, eval_obj, u_val[0, 0],
            #                                                                     u_val[1, 0]))
            # print("    obj_x = {}, obj_u = {}".format(self.obj_x, self.obj_u))

            # record data for the current step
            self.ulast = self.dpin_calc
            self.xlast = dae.x[self.xidx]

        dae.g[self.dpin] = dae.y[self.dpin] - self.dpin0
        dae.g[self.pin] += dae.y[self.dpin]  # positive `dpin` increases the `pin` reference

    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.dpin, self.dpin)
        dae.add_jac(Gy0, 1, self.pin, self.dpin)


class AGCSynVSC(AGCSyn, AGCVSCBase):
    """AGC class that modifies Synchronous pm and VSC pref"""
    def __init__(self, system, name):
        AGCSyn.__init__(self, system, name)
        AGCVSCBase.__init__(self, system, name)
        self._init()

    def init1(self, dae):
        AGCSyn.init1(self, dae)
        AGCVSCBase.init1(self, dae)

    def jac0(self, dae):
        AGCSyn.jac0(self, dae)

    def gcall(self, dae):
        AGCSyn.gcall(self, dae)
        AGCVSCBase.gcall(self, dae)

    def gycall(self, dae):
        AGCSyn.gycall(self, dae)
        AGCVSCBase.gycall(self, dae)

    def fcall(self, dae):
        AGCSyn.fcall(self, dae)


class eAGC(ModelBase):
    def __init__(self, system, name):
        super(eAGC, self).__init__(system, name)
        self._group = 'Control'
        self._data.update({
            'cl': None,
            'tl': 0,
            'Pl': None,
            'BA': None,
        })
        self._descr.update({
            'cl': 'Loss sharing coefficient (vector)',
            'tl': 'Time of generator loss',
            'Pl': 'Loss of power generation in pu (vector)',
            'BA': 'Balancing Area that support the Gen loss',
        })
        self._mandatory.extend(['cl', 'tl', 'Pl', 'BA'])
        self.calls.update({
            'gcall': True,
            'init1': True,
            'jac0': False,
            'fcall': False,
        })
        self._service.extend(['ace', 'en'])
        self._params.extend(['cl', 'tl', 'Pl'])
        self._init()

    def init1(self, dae):
        self.ace = [[]] * self.n
        for idx, item in enumerate(self.BA):
            self.ace[idx] = self.read_data_ext('BArea', field='ace', idx=item)

        self.en = zeros(self.n, 1)

    def switch(self):
        """Switch if time for eAgc has come"""
        t = self.system.dae.t
        for idx in range(0, self.n):
            if t >= self.tl[idx]:
                if self.en[idx] == 0:
                    self.en[idx] = 1
                    logger.info('Extended ACE <{}> activated at t = {}.'.format(self.idx[idx], t))

    def gcall(self, dae):
        self.switch()

        for idx in range(0, self.n):
            dae.g[self.ace[idx]] -= mul(self.en[idx], self.cl[:, idx],
                                        self.Pl[idx])
