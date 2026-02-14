"""
Module for Connectivity Manager.

Handles topology analysis (island detection, slack coverage checks).
Status propagation from buses to connected devices is handled by
the ``set_status`` / ``propagate_init_status`` framework in ``System``.
"""

import logging

from andes.utils.misc import elapsed
from andes.shared import np, sparse, spmatrix

logger = logging.getLogger(__name__)


class ConnMan:
    """
    Connectivity Manager for System.

    Performs topology analysis: island detection, islanded bus detection,
    and slack generator coverage checks.

    Attributes
    ----------
    system : System
        System object to manage the connectivity.
    """

    def __init__(self, system=None):
        self.system = system

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

        if info is True:
            logger.info('Connectivity check completed in %s.', s)
            self.summary()
        else:
            logger.debug('Connectivity check completed in %s.', s)

    def _collect_edges(self):
        """
        Collect from-bus and to-bus address pairs from branch models.

        Uses ``ue.v`` (effective status) so that branches on offline buses
        are correctly excluded from topology analysis.

        Returns
        -------
        tuple of (list, list, list)
            (fr, to, u) where fr/to are bus address indices and u is
            effective online status.
        """
        system = self.system
        fr, to, u = list(), list(), list()

        # TODO: generalize it to all serial devices
        # collect from Line
        fr.extend(system.Line.a1.a.tolist())
        to.extend(system.Line.a2.a.tolist())
        u.extend(system.Line.ue.v.tolist())

        # collect from Jumper
        fr.extend(system.Jumper.a1.a.tolist())
        to.extend(system.Jumper.a2.a.tolist())
        u.extend(system.Jumper.ue.v.tolist())

        # collect from Fortescue
        fr.extend(system.Fortescue.a.a.tolist())
        to.extend(system.Fortescue.aa.a.tolist())
        u.extend(system.Fortescue.ue.v.tolist())

        fr.extend(system.Fortescue.a.a.tolist())
        to.extend(system.Fortescue.ab.a.tolist())
        u.extend(system.Fortescue.ue.v.tolist())

        fr.extend(system.Fortescue.a.a.tolist())
        to.extend(system.Fortescue.ac.a.tolist())
        u.extend(system.Fortescue.ue.v.tolist())

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

        Uses ``ue.v`` (effective status) so that slack generators on
        offline buses are correctly excluded.

        Sets ``Bus.nosw_island`` and ``Bus.msw_island``.
        """
        Bus = self.system.Bus

        if len(Bus.island_sets) == 0:
            return

        slack_bus_uid = Bus.idx2uid(self.system.Slack.bus.v)
        slack_ue = self.system.Slack.ue.v

        for idx, island in enumerate(Bus.island_sets):
            island_set = set(island)
            nosw = 1
            for su, uid in zip(slack_ue, slack_bus_uid):
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
            # collect Slack count and bus info for the warning message
            slack = self.system.Slack
            enabled_buses = [int(b) for b, u in zip(slack.bus.v, slack.ue.v) if u > 0]
            unique_buses = sorted(set(enabled_buses))
            logger.warning('  %d slack generators are enabled on %d bus(es): %s.',
                           len(enabled_buses), len(unique_buses), unique_buses)
            logger.debug("  Bus indices in multiple-Slack areas (0-based): %s",
                         [island_sets[item] for item in msw_island])

        if len(nosw_island) == 0 and len(msw_island) == 0:
            logger.info('  Each island has a slack bus correctly defined and enabled.')
