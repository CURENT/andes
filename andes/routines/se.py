"""
Static and dynamic state estimation routine.
"""

import logging
from collections import OrderedDict

import numpy as np

from andes.routines.base import BaseRoutine
from andes.se.measurement import Measurements, StaticEvaluator
from andes.se.algorithms import wls
from andes.utils.misc import elapsed

logger = logging.getLogger(__name__)


class SE(BaseRoutine):
    """
    State estimation routine.

    Supports static SE (single snapshot, WLS by default) and will support
    dynamic SE (time-stepping with EKF/UKF) in a future phase.

    Static SE estimates bus voltage magnitudes and angles from a set of
    measurements.  Requires a converged power flow as starting point.

    Examples
    --------
    Basic usage with auto-generated measurements::

        ss = andes.load('ieee14.raw')
        ss.PFlow.run()
        ss.SE.run()

    With custom measurements::

        from andes.se import Measurements
        m = Measurements(ss)
        m.add('Bus', 'v', idx=[1, 2, 6, 8], sigma=0.005)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=42)
        ss.SE.run(measurements=m)
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((('tol', 1e-4),
                                      ('max_iter', 20),
                                      ('flat_start', 0),
                                      ('report', 1),
                                      )))
        self.config.add_extra("_help",
                              tol="convergence tolerance",
                              max_iter="max number of WLS iterations",
                              flat_start="use flat start (1.0 pu, 0 rad) as initial guess",
                              report="write output report",
                              )
        self.config.add_extra("_alt",
                              tol="float",
                              max_iter=">=1",
                              flat_start=(0, 1),
                              report=(0, 1),
                              )

        self.converged = False
        self.result = None
        self.measurements = None
        self.evaluator = None

    def summary(self):
        """Print SE configuration summary."""
        out = ['',
               '-> State Estimation',
               f'{"Method":>16s}: WLS',
               f'{"Tolerance":>16s}: {self.config.tol}',
               f'{"Max iterations":>16s}: {self.config.max_iter}',
               ]
        logger.info('\n'.join(out))

    def init(self, measurements=None, algorithm=None):
        """
        Initialize state estimation.

        Parameters
        ----------
        measurements : Measurements or None
            If None, auto-generates measurements from converged PFlow
            with default noise levels.
        algorithm : callable or None
            Estimation algorithm.  Defaults to WLS.
        """
        system = self.system

        if not system.PFlow.converged:
            logger.error("Power flow has not converged. Run PFlow first.")
            return False

        # Set up measurements
        if measurements is not None:
            self.measurements = measurements
        else:
            self.measurements = self._default_measurements()

        # Add reference bus angle constraint if no angle measurements exist
        self._ensure_angle_reference()

        if not self.measurements._finalized:
            self.measurements.finalize()

        # Build evaluator
        self.evaluator = StaticEvaluator(system, self.measurements)

        return True

    def run(self, measurements=None, algorithm=None, **kwargs):
        """
        Run static state estimation.

        Parameters
        ----------
        measurements : Measurements or None
            Measurement data.  If None, auto-generates from PFlow.
        algorithm : callable or None
            Algorithm function with signature
            ``f(evaluator, x0, tol, max_iter) -> dict``.
            Defaults to WLS.

        Returns
        -------
        bool
            Convergence status.
        """
        system = self.system

        self.summary()

        t0, _ = elapsed()

        if not self.init(measurements=measurements, algorithm=algorithm):
            return False

        # Initial state guess
        nb = system.Bus.n
        if self.config.flat_start:
            x0 = np.zeros(2 * nb, dtype=float)
            x0[nb:] = 1.0  # Vm = 1.0 pu
        else:
            # Use PFlow solution as starting point
            x0 = np.concatenate([
                np.array(system.Bus.a.v, dtype=float),
                np.array(system.Bus.v.v, dtype=float),
            ])

        # Run algorithm
        algo = algorithm if algorithm is not None else wls
        self.result = algo(
            self.evaluator, x0,
            tol=self.config.tol,
            max_iter=self.config.max_iter,
        )

        self.converged = self.result['converged']

        t1, s1 = elapsed(t0)
        self.exec_time = t1 - t0

        if self.converged:
            logger.info('SE converged in %d iterations in %s.',
                        self.result['n_iter'], s1)
        else:
            logger.warning('SE did not converge after %d iterations.',
                           self.result['n_iter'])

        if self.config.report:
            self.report()

        system.exit_code = 0 if self.converged else 1
        return self.converged

    def report(self):
        """Log a summary of SE results."""
        if self.result is None:
            return

        nb = self.system.Bus.n
        nm = self.measurements.nm
        r = self.result

        out = ['',
               '-> SE Report',
               f'{"Converged":>16s}: {r["converged"]}',
               f'{"Iterations":>16s}: {r["n_iter"]}',
               f'{"Objective J":>16s}: {r["J"]:.6g}',
               f'{"Measurements":>16s}: {nm}',
               f'{"States":>16s}: {2 * nb}',
               f'{"Redundancy":>16s}: {nm / (2 * nb):.2f}x',
               ]

        if r['converged']:
            v_err = np.abs(self.v_est - np.array(self.system.Bus.v.v))
            a_err = np.abs(self.a_est - np.array(self.system.Bus.a.v))
            out.append(f'{"Max |dV|":>16s}: {np.max(v_err):.6g} pu')
            out.append(f'{"Max |da|":>16s}: {np.max(a_err):.6g} rad')

        logger.info('\n'.join(out))

    def chi_squared_test(self, confidence=0.95):
        """
        Perform chi-squared test on the SE result.

        Returns
        -------
        tuple
            (passed, J, threshold, dof) where dof = nm - n_state.
        """
        from scipy.stats import chi2

        if self.result is None:
            raise RuntimeError("No SE result available. Run SE first.")

        nm = self.measurements.nm
        n_state = 2 * self.system.Bus.n
        dof = nm - n_state

        if dof <= 0:
            logger.warning("Degrees of freedom <= 0 (nm=%d, n_state=%d). "
                           "System may be unobservable.", nm, n_state)
            return (False, self.result['J'], float('inf'), dof)

        threshold = chi2.ppf(confidence, dof)
        passed = self.result['J'] < threshold
        return (passed, self.result['J'], threshold, dof)

    # ------------------------------------------------------------------
    #  Properties for convenient result access
    # ------------------------------------------------------------------

    @property
    def v_est(self):
        """Estimated bus voltage magnitudes."""
        if self.result is None:
            return None
        return self.result['x_est'][self.system.Bus.n:]

    @property
    def a_est(self):
        """Estimated bus voltage angles (radians)."""
        if self.result is None:
            return None
        return self.result['x_est'][:self.system.Bus.n]

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _default_measurements(self, seed=None):
        """Generate a default measurement set from PFlow results."""
        m = Measurements(self.system)
        m.add_bus_voltage(sigma=0.01)
        m.add_bus_injection(sigma_p=0.02, sigma_q=0.03)
        m.generate_from_pflow(seed=seed)
        return m

    def _ensure_angle_reference(self):
        """
        Ensure each island has at least one angle reference measurement.

        For multi-island networks, each connected component needs its own
        angle reference to make the WLS gain matrix non-singular.
        Uses ``Bus.island_sets`` from connectivity analysis.
        """
        system = self.system
        m = self.measurements
        Bus = system.Bus

        # Collect bus UIDs that already have angle measurements
        angle_uids = set()
        for i in range(m.nm):
            if m._kind[i] == 'direct' and m._models[i] == 'Bus' and m._vars[i] == 'a':
                angle_uids.add(Bus.idx2uid(m._idx[i]))

        # Get island topology (list of lists of bus UIDs)
        island_sets = Bus.island_sets
        if len(island_sets) == 0:
            island_sets = [list(range(Bus.n))]

        # Build slack bus UID → bus idx mapping
        slack_uid_to_idx = {}
        if hasattr(system, 'Slack') and system.Slack.n > 0:
            for i in range(system.Slack.n):
                bus_idx = system.Slack.bus.v[i]
                uid = Bus.idx2uid(bus_idx)
                slack_uid_to_idx[uid] = bus_idx

        for island in island_sets:
            island_set = set(island)

            # Skip if this island already has an angle measurement
            if angle_uids & island_set:
                continue

            # Pick a reference bus: prefer the slack bus in this island
            ref_bus = None
            for uid, bus_idx in slack_uid_to_idx.items():
                if uid in island_set:
                    ref_bus = bus_idx
                    break

            if ref_bus is None:
                ref_bus = Bus.idx.v[island[0]]

            logger.info("Adding angle reference at bus %s for island "
                        "with %d buses.", ref_bus, len(island))

            # Use public API to add the pseudo-measurement
            m.add('Bus', 'a', idx=[ref_bus], sigma=1e-6)

            # If z was already generated, append the PFlow angle value
            if m.z is not None and len(m.z) == m.nm - 1:
                pflow_angle = float(Bus.a.v[Bus.idx2uid(ref_bus)])
                m.z = np.append(m.z, pflow_angle)
        # If z hasn't been set yet, it will be set by generate_from_pflow or finalize
