"""
Measurement containers and evaluators for state estimation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def gaussian_noise(sigma, size, rng):
    """Default Gaussian noise: e ~ N(0, sigma^2)."""
    return rng.normal(0, sigma, size)


class Measurements:
    """
    Measurement data container for state estimation.

    Each measurement specifies what is measured (model + variable or a computed
    quantity), where (device idx), the observed value, and its standard
    deviation.

    The core method is ``add()``, which accepts an ANDES model name and
    variable name.  Convenience wrappers like ``add_bus_voltage()`` delegate
    to it.

    Parameters
    ----------
    system : andes.system.System
        The ANDES system instance (must have completed ``setup()``).
    """

    def __init__(self, system):
        self.system = system

        # Parallel lists — one entry per scalar measurement
        self._models = []      # str model name
        self._vars = []        # str variable name
        self._idx = []         # device idx within that model
        self._sigma = []       # standard deviation
        self._kind = []        # 'direct' | 'p_inj' | 'q_inj' | 'p_flow' | 'q_flow'

        # Populated by finalize()
        self.z = None          # (nm,) measured values
        self.sigma = None      # (nm,) standard deviations
        self._finalized = False

    # ------------------------------------------------------------------
    #  Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sigma(sigma, name='sigma'):
        """Raise ValueError if any sigma value is non-positive."""
        arr = np.atleast_1d(np.asarray(sigma, dtype=float))
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be positive.")

    # ------------------------------------------------------------------
    #  Core add method
    # ------------------------------------------------------------------

    # Measurement types supported by StaticEvaluator as direct lookups
    _SUPPORTED_DIRECT = {('Bus', 'v'), ('Bus', 'a')}

    def add(self, model, var, idx=None, sigma=0.01):
        """
        Add measurements of ``model.var`` for selected devices.

        Parameters
        ----------
        model : str
            ANDES model name (e.g. ``'Bus'``, ``'GENROU'``, ``'Line'``).
        var : str
            Variable name within the model (e.g. ``'v'``, ``'a'``, ``'delta'``).
        idx : array-like or None
            Device indices.  ``None`` selects all devices of the model.
        sigma : float or array-like
            Standard deviation(s) for the measurements.  Must be positive.
        """
        mdl = self.system.__dict__.get(model)
        if mdl is None:
            raise ValueError(f"Model '{model}' not found in system.")

        if not hasattr(mdl, var):
            raise ValueError(f"Variable '{var}' not found on model '{model}'.")

        if (model, var) not in self._SUPPORTED_DIRECT:
            raise ValueError(
                f"Direct measurement '{model}.{var}' is not supported by "
                f"StaticEvaluator. Supported: {self._SUPPORTED_DIRECT}."
            )

        if idx is None:
            idx = list(mdl.idx.v)
        elif not hasattr(idx, '__len__'):
            idx = [idx]
        else:
            idx = list(idx)

        sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (len(idx),))
        self._check_sigma(sigma_arr)

        for i, dev_idx in enumerate(idx):
            self._models.append(model)
            self._vars.append(var)
            self._idx.append(dev_idx)
            self._sigma.append(float(sigma_arr[i]))
            self._kind.append('direct')

        self._finalized = False

    # ------------------------------------------------------------------
    #  Convenience wrappers
    # ------------------------------------------------------------------

    def add_bus_voltage(self, bus_idx=None, sigma=0.01):
        """Add bus voltage magnitude measurements."""
        self.add('Bus', 'v', idx=bus_idx, sigma=sigma)

    def add_bus_angle(self, bus_idx=None, sigma=0.01):
        """Add bus voltage angle measurements (PMU)."""
        self.add('Bus', 'a', idx=bus_idx, sigma=sigma)

    def add_bus_injection(self, bus_idx=None, sigma_p=0.02, sigma_q=0.03):
        """Add active and reactive power injection measurements.

        These are 'computed' measurements evaluated via Y-bus formulas,
        not direct DAE variable lookups.
        """
        self._check_sigma(sigma_p, 'sigma_p')
        self._check_sigma(sigma_q, 'sigma_q')

        bus = self.system.Bus
        if bus_idx is None:
            bus_idx = list(bus.idx.v)
        elif not hasattr(bus_idx, '__len__'):
            bus_idx = [bus_idx]
        else:
            bus_idx = list(bus_idx)

        for bidx in bus_idx:
            self._models.append('Bus')
            self._vars.append('p_inj')
            self._idx.append(bidx)
            self._sigma.append(float(sigma_p))
            self._kind.append('p_inj')

            self._models.append('Bus')
            self._vars.append('q_inj')
            self._idx.append(bidx)
            self._sigma.append(float(sigma_q))
            self._kind.append('q_inj')

        self._finalized = False

    def add_line_flow(self, line_idx=None, sigma_p=0.02, sigma_q=0.03):
        """Add active and reactive power flow measurements (from-end)."""
        self._check_sigma(sigma_p, 'sigma_p')
        self._check_sigma(sigma_q, 'sigma_q')

        line = self.system.Line
        if line_idx is None:
            line_idx = list(line.idx.v)
        elif not hasattr(line_idx, '__len__'):
            line_idx = [line_idx]
        else:
            line_idx = list(line_idx)

        for lidx in line_idx:
            self._models.append('Line')
            self._vars.append('p_flow')
            self._idx.append(lidx)
            self._sigma.append(float(sigma_p))
            self._kind.append('p_flow')

            self._models.append('Line')
            self._vars.append('q_flow')
            self._idx.append(lidx)
            self._sigma.append(float(sigma_q))
            self._kind.append('q_flow')

        self._finalized = False

    # ------------------------------------------------------------------
    #  Finalize and generate values
    # ------------------------------------------------------------------

    def finalize(self):
        """Convert internal lists to arrays and resolve addresses."""
        self.sigma = np.array(self._sigma, dtype=float)
        if self.z is None:
            self.z = np.zeros(self.nm, dtype=float)
        self._finalized = True

    def generate_from_pflow(self, noise_func=None, seed=None):
        """
        Set measurement values from the converged power flow solution.

        Computes ``z = h(x_true) + noise``.

        Parameters
        ----------
        noise_func : callable or None
            ``noise_func(sigma, size, rng) -> array``.
            Defaults to Gaussian noise.
        seed : int or None
            Random seed for reproducibility.
        """
        if not self.system.PFlow.converged:
            logger.warning("Power flow has not converged. Measurement values "
                           "from generate_from_pflow() may be invalid.")

        if not self._finalized:
            self.finalize()

        if noise_func is None:
            noise_func = gaussian_noise

        rng = np.random.default_rng(seed)

        # Build evaluator to compute h(x_true)
        evaluator = StaticEvaluator(self.system, self)
        theta_true = np.array(self.system.Bus.a.v, dtype=float)
        Vm_true = np.array(self.system.Bus.v.v, dtype=float)
        h_true = evaluator.h(theta_true, Vm_true)

        noise = noise_func(self.sigma, self.nm, rng)
        self.z = h_true + noise

    @property
    def nm(self):
        """Number of measurements."""
        return len(self._models)


class StaticEvaluator:
    """
    Evaluate measurement functions h(x) and Jacobian H(x) for static SE.

    Uses the Y-bus matrix for power injection and flow calculations.
    Direct variable measurements (voltage magnitude, angle) are simple
    lookups.

    Parameters
    ----------
    system : andes.system.System
    measurements : Measurements
    """

    def __init__(self, system, measurements):
        self.system = system
        self.meas = measurements
        self.nb = system.Bus.n

        # Build Y-bus and convert to dense numpy
        Y_sparse = system.Line.build_y()
        self.Y = np.zeros((self.nb, self.nb), dtype=complex)
        # Extract triplets from kvxopt spmatrix
        _triplets = Y_sparse.CCS
        col_ptr, row_idx, vals = _triplets
        for col in range(self.nb):
            for k in range(col_ptr[col], col_ptr[col + 1]):
                r = row_idx[k]
                self.Y[r, col] = complex(vals[k])

        # Add shunt admittances from Shunt model if present
        if hasattr(system, 'Shunt') and system.Shunt.n > 0:
            shunt = system.Shunt
            for i in range(shunt.n):
                if shunt.u.v[i] == 0:
                    continue
                bus_uid = system.Bus.idx2uid(shunt.bus.v[i])
                self.Y[bus_uid, bus_uid] += complex(shunt.g.v[i], shunt.b.v[i])

        # Build bus idx → uid map
        self._bus_idx2uid = {}
        for uid, idx in enumerate(system.Bus.idx.v):
            self._bus_idx2uid[idx] = uid

        # Build line idx → (from_uid, to_uid, y_series, y_sh, tap, phi) map
        self._line_params = {}
        line = system.Line
        for i in range(line.n):
            lidx = line.idx.v[i]
            from_uid = self._bus_idx2uid[line.bus1.v[i]]
            to_uid = self._bus_idx2uid[line.bus2.v[i]]
            y_series = line.u.v[i] / complex(line.r.v[i], line.x.v[i])
            y_sh = line.u.v[i] * complex(line.g.v[i], line.b.v[i]) / 2
            y1 = line.u.v[i] * complex(line.g1.v[i], line.b1.v[i])
            y2 = line.u.v[i] * complex(line.g2.v[i], line.b2.v[i])
            tap = line.tap.v[i]
            phi = line.phi.v[i]
            self._line_params[lidx] = (from_uid, to_uid, y_series, y_sh, y1, y2, tap, phi)

        # Pre-classify measurements by kind for fast evaluation
        self._direct_v_uid = []     # (meas_pos, bus_uid) for v_mag
        self._direct_a_uid = []     # (meas_pos, bus_uid) for v_ang
        self._pinj_uid = []         # (meas_pos, bus_uid) for p_inj
        self._qinj_uid = []         # (meas_pos, bus_uid) for q_inj
        self._pflow_params = []     # (meas_pos, line_param_tuple)
        self._qflow_params = []     # (meas_pos, line_param_tuple)

        meas = measurements
        for i in range(meas.nm):
            kind = meas._kind[i]
            if kind == 'direct':
                if meas._models[i] == 'Bus' and meas._vars[i] == 'v':
                    uid = self._bus_idx2uid[meas._idx[i]]
                    self._direct_v_uid.append((i, uid))
                elif meas._models[i] == 'Bus' and meas._vars[i] == 'a':
                    uid = self._bus_idx2uid[meas._idx[i]]
                    self._direct_a_uid.append((i, uid))
            elif kind == 'p_inj':
                uid = self._bus_idx2uid[meas._idx[i]]
                self._pinj_uid.append((i, uid))
            elif kind == 'q_inj':
                uid = self._bus_idx2uid[meas._idx[i]]
                self._qinj_uid.append((i, uid))
            elif kind == 'p_flow':
                params = self._line_params[meas._idx[i]]
                self._pflow_params.append((i, params))
            elif kind == 'q_flow':
                params = self._line_params[meas._idx[i]]
                self._qflow_params.append((i, params))

    def h(self, theta, Vm):
        """
        Evaluate all measurement functions.

        Parameters
        ----------
        theta : ndarray, shape (nb,)
            Bus voltage angles in radians.
        Vm : ndarray, shape (nb,)
            Bus voltage magnitudes in per unit.

        Returns
        -------
        ndarray, shape (nm,)
            Computed measurement values.
        """
        nm = self.meas.nm
        hx = np.zeros(nm, dtype=float)

        # Complex voltage
        V = Vm * np.exp(1j * theta)

        # Direct voltage magnitude
        for pos, uid in self._direct_v_uid:
            hx[pos] = Vm[uid]

        # Direct voltage angle
        for pos, uid in self._direct_a_uid:
            hx[pos] = theta[uid]

        # Power injections: S_i = V_i * conj(sum_j Y_ij V_j)
        if self._pinj_uid or self._qinj_uid:
            S_inj = V * np.conj(self.Y @ V)
            for pos, uid in self._pinj_uid:
                hx[pos] = S_inj[uid].real
            for pos, uid in self._qinj_uid:
                hx[pos] = S_inj[uid].imag

        # Line flows (from-end): S_ij = V_i * conj(I_ij)
        # I_ij = (V_i/tap^2 - V_j/(tap*exp(-j*phi))) * y_series + V_i/tap^2 * (y_sh + y1)
        for pos, (fi, ti, ys, ysh, y1, y2, tap, phi) in self._pflow_params:
            m = tap * np.exp(1j * phi)
            I_ij = (V[fi] / (tap**2) - V[ti] / np.conj(m)) * ys + V[fi] / (tap**2) * (ysh + y1)
            hx[pos] = (V[fi] * np.conj(I_ij)).real

        for pos, (fi, ti, ys, ysh, y1, y2, tap, phi) in self._qflow_params:
            m = tap * np.exp(1j * phi)
            I_ij = (V[fi] / (tap**2) - V[ti] / np.conj(m)) * ys + V[fi] / (tap**2) * (ysh + y1)
            hx[pos] = (V[fi] * np.conj(I_ij)).imag

        return hx

    def H_numerical(self, theta, Vm, eps=1e-5):
        """
        Numerical Jacobian of h(x) via central differences.

        Parameters
        ----------
        theta : ndarray, shape (nb,)
        Vm : ndarray, shape (nb,)
        eps : float
            Perturbation size.

        Returns
        -------
        ndarray, shape (nm, 2*nb)
            Jacobian matrix.  Columns: [dh/dθ_1..dh/dθ_nb, dh/dV_1..dh/dV_nb].
        """
        nb = self.nb
        nm = self.meas.nm
        H = np.zeros((nm, 2 * nb), dtype=float)

        # Perturb theta columns
        for j in range(nb):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[j] += eps
            theta_m[j] -= eps
            H[:, j] = (self.h(theta_p, Vm) - self.h(theta_m, Vm)) / (2 * eps)

        # Perturb Vm columns
        for j in range(nb):
            Vm_p = Vm.copy()
            Vm_m = Vm.copy()
            Vm_p[j] += eps
            Vm_m[j] -= eps
            H[:, nb + j] = (self.h(theta, Vm_p) - self.h(theta, Vm_m)) / (2 * eps)

        return H

    def residual(self, theta, Vm):
        """Measurement residual: z - h(x)."""
        return self.meas.z - self.h(theta, Vm)

    def weight_matrix(self):
        """Diagonal weight matrix W = diag(1/sigma^2)."""
        return np.diag(1.0 / self.meas.sigma ** 2)
