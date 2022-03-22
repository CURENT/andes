"""
[WIP] Interoperability with pypowsybl.
"""

import logging
import pypowsybl as pp

logger = logging.getLogger(__name__)


def to_pypowsybl(ss):
    """
    Convert an ANDES system to a pypowsybl network.

    Parameters
    ----------
    ss : andes.system.System
        The ANDES system to be converted.

    Returns
    -------
    pypowsybl.network.Network

    """

    net = pp.network.create_empty()

    pass