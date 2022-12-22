"""
Basic GridCal (4.6.1) interface, based on the pandapower interface written by Jinning Wang

Author: Josep Fanals (@JosepFanals)
"""

import logging
from functools import wraps

import numpy as np
import matplotlib

from andes.shared import GridCal_Engine as gc

matplotlib.use('agg')
logger = logging.getLogger(__name__)


def require_gridcal(f):
    """
    Decorator for functions that require GridCal.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            getattr(gc, '__name__')
        except AttributeError as exc:
            raise ModuleNotFoundError("GridCal needs to be installed manually.") from exc

        return f(*args, **kwds)

    return wrapper


def _to_gc_bus(ssp, ssa_bus):
    """
    Define buses in GridCal's grid.
    """

    dic_bus = {}

    for i in range(ssa_bus.n):
        bus = gc.Bus(name=ssa_bus.name.v[i],
                     active=ssa_bus.u.v[i],
                     vnom=ssa_bus.Vn.v[i],
                     vmin=ssa_bus.vmin.v[i],
                     vmax=ssa_bus.vmax.v[i],
                     xpos=ssa_bus.xcoord.v[i],
                     ypos=ssa_bus.ycoord.v[i],
                     Vm0=ssa_bus.v0.v[i],
                     Va0=ssa_bus.a0.v[i])

        dic_bus[ssa_bus.idx.v[i]] = bus

        ssp.add_bus(bus)

    return ssp, dic_bus


def _to_gc_branch(ssp, ssa_line, dic_bus):
    """
    Define branches (lines and trafos) in GridCal's grid.
    """

    rate = [xx if xx != 0.0 else 1.0 for xx in ssa_line.rate_a.v]  # 1.0 is the default in GridCal

    for i in range(len(ssa_line)):

        if ssa_line.tap.v[i] != 1.0 or ssa_line.phi.v[i] != 0.0:
            # trafo
            trafo = gc.Transformer2W(bus_from=dic_bus[ssa_line.bus1.v[i]],
                                     bus_to=dic_bus[ssa_line.bus2.v[i]],
                                     name=ssa_line.name.v[i],
                                     active=ssa_line.u.v[i],
                                     r=ssa_line.r.v[i],
                                     x=ssa_line.x.v[i],
                                     b=ssa_line.b.v[i],
                                     g=ssa_line.g.v[i],
                                     rate=rate[i],
                                     tap=ssa_line.tap.v[i],
                                     shift_angle=ssa_line.phi.v[i])

            ssp.add_transformer2w(trafo)

        else:
            # line
            line = gc.Line(bus_from=dic_bus[ssa_line.bus1.v[i]],
                           bus_to=dic_bus[ssa_line.bus2.v[i]],
                           name=ssa_line.name.v[i],
                           active=ssa_line.u.v[i],
                           r=ssa_line.r.v[i],
                           x=ssa_line.x.v[i],
                           b=ssa_line.b.v[i],
                           rate=rate[i])

            ssp.add_line(line)

    return ssp


def _to_gc_load(ssp, ssa_load, dic_bus, sbase=1.0):
    """
    Define loads in GridCal's grid.
    """

    for i in range(len(ssa_load)):
        load = gc.Load(name=ssa_load.name.v[i],
                       active=ssa_load.u.v[i],
                       P=ssa_load.p0.v[i] * sbase,
                       Q=ssa_load.q0.v[i] * sbase)

        ssp.add_load(dic_bus[ssa_load.bus.v[i]], load)

    return ssp


def _to_gc_shunt(ssp, ssa_shunt, dic_bus, sbase=1.0):
    """
    Define shunts in GridCal's grid.
    """

    for i in range(len(ssa_shunt)):
        shunt = gc.Shunt(name=ssa_shunt.name.v[i],
                         active=ssa_shunt.u.v[i],
                         G=ssa_shunt.g.v[i] * sbase,
                         B=ssa_shunt.b.v[i] * sbase)

        ssp.add_shunt(dic_bus[ssa_shunt.bus.v[i]], shunt)

    return ssp


def _to_gc_generator(ssp, ssa_slack, ssa_pv, dic_bus, sbase=1.0):
    """
    Define generators considering slack and PV buses.
    """

    for i in range(len(ssa_slack)):

        sl_name = dic_bus[ssa_slack.bus.v[i]]
        b2_dict = ssp.get_bus_names()
        bus_id = b2_dict.index(sl_name.name)
        ssp.buses[bus_id].is_slack = True

        gen = gc.Generator(name=str(ssa_slack.SynGen.v[i]),
                           active=ssa_slack.u.v[i],
                           active_power=ssa_slack.p0.v[i] * sbase,
                           power_factor=ssa_slack.p0.v[i] / np.sqrt((ssa_slack.p0.v[i]**2 + ssa_slack.q0.v[i]**2)),
                           p_min=ssa_slack.pmin.v[i] * sbase,
                           p_max=ssa_slack.pmax.v[i] * sbase,
                           Qmin=ssa_slack.qmin.v[i] * sbase,
                           Qmax=ssa_slack.qmax.v[i] * sbase,
                           voltage_module=ssa_slack.v0.v[i],
                           Snom=ssa_slack.Sn.v[i])

        ssp.add_generator(dic_bus[ssa_slack.bus.v[i]], gen)

    for i in range(len(ssa_pv)):
        gen = gc.Generator(name=str(ssa_pv.SynGen.v[i]),
                           active=ssa_pv.u.v[i],
                           active_power=ssa_pv.p0.v[i] * sbase,
                           power_factor=ssa_pv.p0.v[i] / np.sqrt((ssa_pv.p0.v[i]**2 + ssa_pv.q0.v[i]**2)),
                           p_min=ssa_pv.pmin.v[i] * sbase,
                           p_max=ssa_pv.pmax.v[i] * sbase,
                           Qmin=ssa_pv.qmin.v[i] * sbase,
                           Qmax=ssa_pv.qmax.v[i] * sbase,
                           voltage_module=ssa_pv.v0.v[i],
                           Snom=ssa_pv.Sn.v[i])

        ssp.add_generator(dic_bus[ssa_pv.bus.v[i]], gen)

    return ssp


@require_gridcal
def to_gridcal(ssa, verify=True, tol=1e-6):
    """
    Convert an ANDES system to a GridCal grid.

    Parameters
    ----------
    ssa : andes.system.System
        The ANDES system to be converted
    verify : bool
        If True, the converted network will be verified with the source ANDES
        system using AC power flow.
    tol : float
        The tolerance of error when comparing power flow solutions.

    Returns
    -------
    GridCal.Engine.Core.multi_circuit.MultiCircuit
        A GridCal net with the same bus, branch, gen, and load data as the ANDES
        system

    Notes
    -----
    Handling of the following parameters:

      - By default, all generators in ``ssp`` are controllable unless
        user-defined controllability is given
      - The online status of generators are determined by the online status of
        ``StaticGen`` that connected to the ``SynGen`` or ``DG``
      - ``ssp.gen.name`` is from ``ssa.StaticGen.idx``, which should be unique

    """

    # create an empty GC grid
    sbase = ssa.config.mva
    ssp = gc.MultiCircuit(Sbase=sbase, fbase=ssa.config.freq, name=ssa.name)

    # 1. convert buses
    ssp, dic_bus = _to_gc_bus(ssp, ssa.Bus)

    # 2. convert branches
    ssp = _to_gc_branch(ssp, ssa.Line, dic_bus)

    # 3. convert loads
    ssp = _to_gc_load(ssp, ssa.PQ, dic_bus, sbase=sbase)

    # 4. convert shunts
    ssp = _to_gc_shunt(ssp, ssa.Shunt, dic_bus, sbase=sbase)

    # 5. convert generators (Slack and PV)
    ssp = _to_gc_generator(ssp, ssa.Slack, ssa.PV, dic_bus, sbase=sbase)

    if verify:
        _verify_pf(ssa, ssp, tol)

    return ssp


def _verify_pf(ssa, ssp, tol=1e-6):
    """
    Verify power flow results.
    """

    # ANDES
    ssa.PFlow.run()
    pf_bus = ssa.Bus.as_df()[["name"]]

    pf_bus['v_andes'] = ssa.Bus.v.v
    pf_bus['a_andes'] = ssa.Bus.a.v

    # GridCal
    options = gc.PowerFlowOptions(gc.SolverType.NR, verbose=False)
    pf = gc.PowerFlowDriver(ssp, options)
    pf.run()

    # GridCal assumes the slack has an angle of 0 always, correct it
    pf.results.voltage = pf.results.voltage * np.exp(1j * ssa.Slack.a0.v[0])

    # Check
    vm_dif = pf_bus['v_andes'] - np.abs(pf.results.voltage)
    va_dif = pf_bus['a_andes'] - np.angle(pf.results.voltage)

    ret = False
    if (np.max(np.abs(vm_dif)) < tol) and (np.max(np.abs(va_dif)) < tol):
        logger.info("Power flow results are consistent. Conversion is successful.")
        ret = True
    else:
        logger.warning("Warning: Power flow results are inconsistent. Please check!")
        logger.warning(vm_dif)
        logger.warning(va_dif)

    return ret
