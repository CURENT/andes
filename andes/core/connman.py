"""
Module for Connectivity Manager.
"""

import logging
from collections import OrderedDict

from andes.utils.func import list_flatten
from andes.utils.misc import elapsed
from andes.shared import np, sparse, spmatrix

logger = logging.getLogger(__name__)


# connectivity dependencies of `Bus`
# NOTE: only include PFlow models and measurements models
# cause online status of dynamic models are expected to be handled by their
# corresponding static models
# TODO: DC Topologies are not included yet, `Node`, etc
bus_deps = OrderedDict([
    ('ACLine', ['bus1', 'bus2']),
    ('ACShort', ['bus1', 'bus2']),
    ('FreqMeasurement', ['bus']),
    ('Interface', ['bus']),
    ('Motor', ['bus']),
    ('PhasorMeasurement', ['bus']),
    ('StaticACDC', ['bus']),
    ('StaticGen', ['bus']),
    ('StaticLoad', ['bus']),
    ('StaticShunt', ['bus']),
])


class ConnMan:
    """
    Define a Connectivity Manager class for System.

    Connectivity Manager handles both bus on/off state tracking and
    topology analysis (island detection, slack coverage checks).

    Attributes
    ----------
    system: system object
        System object to manage the connectivity.
    busu0: ndarray
        Last recorded bus connection status.
    is_needed: bool
        Flag to indicate if connectivity update is needed.
    changes: dict
        Dictionary to record bus connectivity changes ('on' and 'off').
        'on' means the bus is previous offline and now online.
        'off' means the bus is previous online and now offline.
    """

    def __init__(self, system=None):
        """
        Initialize the connectivity manager.

        Parameters
        ----------
        system: system object
            System object to manage the connectivity.
        """
        self.system = system
        self.busu0 = None               # placeholder for Bus.u.v
        self.is_needed = False          # flag to indicate if check is needed
        self.changes = {'on': None, 'off': None}    # dict of bus connectivity changes

    def init(self):
        """
        Initialize the connectivity.

        `ConnMan` is initialized in `System.setup()`, where all buses are considered online
        by default. This method records the initial bus connectivity.
        """
        # NOTE: here, we expect all buses are online before the system setup
        self.busu0 = np.ones(self.system.Bus.n, dtype=int)
        self.changes['on'] = np.zeros(self.system.Bus.n, dtype=int)
        self.changes['off'] = np.logical_and(self.busu0 == 1, self.system.Bus.u.v == 0).astype(int)

        if np.any(self.changes['off']):
            self.is_needed = True

        self.act()

        return True

    def _update(self):
        """
        Helper function for in-place update of bus connectivity.
        """
        self.changes['on'][...] = np.logical_and(self.busu0 == 0, self.system.Bus.u.v == 1)
        self.changes['off'][...] = np.logical_and(self.busu0 == 1, self.system.Bus.u.v == 0)
        self.busu0[...] = self.system.Bus.u.v

    def record(self):
        """
        Record the bus connectivity in-place.

        This method should be called if `Bus.set()` or `Bus.alter()` is called.
        """
        self._update()

        if np.any(self.changes['on']):
            onbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["on"])[0]]
            logger.warning(f'Bus turned on: {onbus_idx}')
            self.is_needed = True
            if len(onbus_idx) > 0:
                raise NotImplementedError('Turning on bus after system setup is not supported yet!')

        if np.any(self.changes['off']):
            offbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["off"])[0]]
            logger.warning(f'Bus turned off: {offbus_idx}')
            self.is_needed = True

        return self.changes

    def act(self):
        """
        Update the connectivity.
        """
        if not self.is_needed:
            logger.debug('No need to update connectivity.')
            return True

        if self.system.TDS.initialized:
            raise NotImplementedError('Bus connectivity update during TDS is not supported yet!')

        # --- action ---
        offbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["off"])[0]]

        # skip if no bus is turned off
        if len(offbus_idx) == 0:
            return True

        logger.warning('Entering connectivity update.')
        logger.warning(f'Following bus(es) are turned off: {offbus_idx}')

        logger.warning('-> System connectivity update results:')
        for grp_name, src_list in bus_deps.items():
            devices = []
            for src in src_list:
                grp_devs = self.system.__dict__[grp_name].find_idx(keys=src, values=offbus_idx,
                                                                   allow_none=True, allow_all=True,
                                                                   default=None)
                grp_devs_flat = list_flatten(grp_devs)
                if grp_devs_flat != [None]:
                    devices.append(grp_devs_flat)

            devices_flat = list_flatten(devices)

            if len(devices_flat) > 0:
                self.system.__dict__[grp_name].set(src='u', attr='v',
                                                   idx=devices_flat, value=0)
                logger.warning(f'In <{grp_name}>, turn off {devices_flat}')

        self.is_needed = False      # reset the action flag
        self._update()              # update but not record
        self.check_connectivity(info=True)
        return True

    # ------------------------------------------------------------------
    #  Topology analysis
    # ------------------------------------------------------------------

    def check_connectivity(self, info=True):
        """
        Perform connectivity check for the system.

        Parameters
        ----------
        info : bool
            True to log connectivity summary.
        """
        t0, _ = elapsed()
        logger.debug("Entering connectivity check.")

        system = self.system
        Bus = system.Bus

        Bus.n_islanded_buses = 0
        Bus.islanded_buses = list()
        Bus.island_sets = list()
        Bus.nosw_island = list()
        Bus.msw_island = list()
        Bus.islands = list()

        n = Bus.n

        fr, to, u = self._collect_edges()
        self._find_islanded_buses(n, fr, to, u)
        self._find_islands(n, fr, to, u)
        self._check_slack_coverage()
        self._post_process_islands(n)

        _, s = elapsed(t0)
        logger.info('Connectivity check completed in %s.', s)

        if info is True:
            self.summary()

    def _collect_edges(self):
        """
        Collect from-bus and to-bus address pairs from branch models.

        Returns
        -------
        tuple of (list, list, list)
            (fr, to, u) where fr/to are bus address indices and u is online status.
        """
        system = self.system
        fr, to, u = list(), list(), list()

        # TODO: generalize it to all serial devices
        # collect from Line
        fr.extend(system.Line.a1.a.tolist())
        to.extend(system.Line.a2.a.tolist())
        u.extend(system.Line.u.v.tolist())

        # collect from Jumper
        fr.extend(system.Jumper.a1.a.tolist())
        to.extend(system.Jumper.a2.a.tolist())
        u.extend(system.Jumper.u.v.tolist())

        # collect from Fortescue
        fr.extend(system.Fortescue.a.a.tolist())
        to.extend(system.Fortescue.aa.a.tolist())
        u.extend(system.Fortescue.u.v.tolist())

        fr.extend(system.Fortescue.a.a.tolist())
        to.extend(system.Fortescue.ab.a.tolist())
        u.extend(system.Fortescue.u.v.tolist())

        fr.extend(system.Fortescue.a.a.tolist())
        to.extend(system.Fortescue.ac.a.tolist())
        u.extend(system.Fortescue.u.v.tolist())

        return fr, to, u

    def _find_islanded_buses(self, n, fr, to, u):
        """
        Find buses with zero active connections (degree-zero nodes).

        Sets ``Bus.islanded_buses``, ``Bus.islanded_a``,
        ``Bus.islanded_v``, and ``Bus.n_islanded_buses``.
        """
        Bus = self.system.Bus

        os = [0] * len(u)

        # find islanded buses from sparse degree vector (avoid dense allocation)
        degree_sp = sparse(spmatrix(u, to, os, (n, 1), 'd') +
                           spmatrix(u, fr, os, (n, 1), 'd'))
        connected = set(int(i) for i in degree_sp.I)
        Bus.islanded_buses = [i for i in range(n) if i not in connected]

        # store `a` and `v` indices for zeroing out residuals
        Bus.islanded_a = np.array(Bus.islanded_buses)
        Bus.islanded_v = Bus.n + Bus.islanded_a
        Bus.n_islanded_buses = len(Bus.islanded_a)

    def _find_islands(self, n, fr, to, u):
        """
        Find connected components via iterative DFS — O(n + edges).

        Sets ``Bus.island_sets``.
        """
        Bus = self.system.Bus

        adj = [[] for _ in range(n)]
        for f, t, status in zip(fr, to, u):
            if status != 0:
                adj[f].append(t)
                adj[t].append(f)

        visited = np.zeros(n, dtype=bool)
        if Bus.n_islanded_buses > 0:
            visited[Bus.islanded_a] = True

        island_sets = []
        for bus in range(n):
            if visited[bus]:
                continue
            stack = [bus]
            visited[bus] = True
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for nbr in adj[node]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        stack.append(nbr)
            island_sets.append(component)

        Bus.island_sets = island_sets

    def _check_slack_coverage(self):
        """
        Check if each island has exactly one slack generator.

        Sets ``Bus.nosw_island`` and ``Bus.msw_island``.
        """
        Bus = self.system.Bus

        if len(Bus.island_sets) == 0:
            return

        slack_bus_uid = Bus.idx2uid(self.system.Slack.bus.v)
        slack_u = self.system.Slack.u.v

        for idx, island in enumerate(Bus.island_sets):
            island_set = set(island)
            nosw = 1
            for su, uid in zip(slack_u, slack_bus_uid):
                if (su == 1) and (uid in island_set):
                    nosw -= 1
            if nosw == 1:
                Bus.nosw_island.append(idx)
            elif nosw < 0:
                Bus.msw_island.append(idx)

    def _post_process_islands(self, n):
        """
        Build the unified ``Bus.islands`` list and identify the largest
        island for generator criteria checks during TDS.
        """
        system = self.system
        Bus = system.Bus

        # 1. extend islanded buses, each in a list
        if len(Bus.islanded_buses) > 0:
            Bus.islands.extend([[item] for item in Bus.islanded_buses])

        if len(Bus.island_sets) == 0:
            Bus.islands.append(list(range(n)))
        else:
            Bus.islands.extend(Bus.island_sets)

        # 2. find generators in the largest island
        if system.TDS.config.criteria and system.TDS.initialized:
            lg_island = max(Bus.islands, key=len)

            lg_bus_idx = [Bus.idx.v[ii] for ii in lg_island]
            if system.SynGen.n > 0:
                system.SynGen.store_idx_island(lg_bus_idx)

    def summary(self):
        """
        Print out connectivity check summary.
        """
        Bus = self.system.Bus

        island_sets = Bus.island_sets
        nosw_island = Bus.nosw_island
        msw_island = Bus.msw_island
        n_islanded_buses = Bus.n_islanded_buses

        logger.info("-> System connectivity check results:")
        if n_islanded_buses == 0:
            logger.info("  No islanded bus detected.")
        else:
            logger.info("  %d islanded bus detected.", n_islanded_buses)
            logger.debug("  Islanded Bus indices (0-based): %s", Bus.islanded_buses)

        if len(island_sets) == 0:
            logger.info("  No island detected.")
        elif len(island_sets) == 1:
            logger.info("  System is interconnected.")
            logger.debug("  Bus indices in interconnected system (0-based): %s", island_sets)
        else:
            logger.info("  System contains %d island(s).", len(island_sets))
            logger.debug("  Bus indices in islanded areas (0-based): %s", island_sets)

        if len(nosw_island) > 0:
            logger.warning('  Slack generator is not defined/enabled for %d island(s).',
                           len(nosw_island))
            logger.debug("  Bus indices in no-Slack areas (0-based): %s",
                         [island_sets[item] for item in nosw_island])

        if len(msw_island) > 0:
            logger.warning('  Multiple slack generators are defined/enabled for %d island(s).',
                           len(msw_island))
            logger.debug("  Bus indices in multiple-Slack areas (0-based): %s",
                         [island_sets[item] for item in msw_island])

        if len(nosw_island) == 0 and len(msw_island) == 0:
            logger.info('  Each island has a slack bus correctly defined and enabled.')
