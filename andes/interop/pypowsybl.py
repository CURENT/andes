"""
Interoperability with pypowsybl.
"""

import logging
import numpy as np
from functools import wraps

from andes.shared import pd
from andes.shared import pypowsybl as pp

logger = logging.getLogger(__name__)


def require_pypowsybl(f):
    """
    Decorator for functions that require pypowsybl.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            getattr(pp, '__version__')
        except AttributeError:
            raise ModuleNotFoundError("pypowsybl needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


@require_pypowsybl
def to_pypowsybl(ss):
    """
    Convert an ANDES system to a pypowsybl network.

    Returns
    -------
    pypowsybl.network.Network

    Parameters
    ----------
    ss : andes.system.System
        The ANDES system to be converted.

    Notes
    -----
    - Only the BUS_BREAKER topology is supported.
    - Each bus has a voltage level named "VL" followed by the bus idx.
    - Buses connected by transformers are in the same substation.

    Warnings
    --------
    - Power flow results are not verified.

    Examples
    --------
    One can utilize pypowsybl to draw network topology. For example,

    .. code:: python

        ss = andes.system.example()
        n = to_pypowsybl(ss)
        results = pp.loadflow.run_ac(n)

        n.get_network_area_diagram()  # show diagram for system
        n.get_single_line_diagram("VL6")  # show single-line diagram for bus 6

    """

    substations, voltage_levels = _make_substation_voltage(ss)
    buses = _make_buses(ss)
    lines, transformers = _make_tline_tf(ss)
    loads = _make_loads(ss)
    generators = _make_generators(ss)
    shunt_df, shunt_model_df = _make_shunts(ss)

    n = pp.network.create_empty()
    n.create_substations(substations)
    n.create_voltage_levels(voltage_levels)
    n.create_buses(buses)
    n.create_lines(lines)
    n.create_loads(loads)
    n.create_generators(generators)
    n.create_shunt_compensators(shunt_df, non_linear_model_df=shunt_model_df)
    n.create_2_windings_transformers(transformers)

    return n


def _make_substation_voltage(ss):
    """
    Make dataframes for substation and voltage levels.

    """
    # --- get `idx` for transmission lines (tline) and transformers (tf)
    tf_idx = ss.Line.get_tf_idx()
    bus1_tf = ss.Line.get("bus1", tf_idx)
    bus2_tf = ss.Line.get("bus2", tf_idx)

    # buses attached to the same transformer will be in one substation

    # -- map bus idx (number or string) -> substation idx (number or string)

    bus2subs = {bus_idx: f"S{bus_idx}" for bus_idx in ss.Bus.idx.v}
    in_substation = {}

    # point buses to substations based on transformers
    for bus1, bus2 in zip(bus1_tf, bus2_tf):
        if (bus1 not in in_substation) and (bus2 not in in_substation):
            bus2subs[bus2] = bus2subs[bus1]
            in_substation[bus1] = in_substation[bus2] = bus2subs[bus1]
        elif (bus1 in in_substation):
            bus2subs[bus2] = bus2subs[bus1] = in_substation[bus1]
            in_substation[bus2] = bus2subs[bus1]
        elif (bus2 in in_substation):
            bus2subs[bus2] = bus2subs[bus1] = in_substation[bus2]
            in_substation[bus1] = bus2subs[bus1]

    # one substation can have multiple voltage levels --- map voltage level
    # (number) -> substation name (string) ---
    vl2subs = {bus_idx: f"S{bus_idx}" for bus_idx in ss.Bus.idx.v}

    # update voltage levels of `bus2` to point to substation named after `bus1`
    for bus1, bus2 in zip(bus1_tf, bus2_tf):
        vl2subs[bus1] = bus2subs[bus1]
        vl2subs[bus2] = bus2subs[bus2]

    substation_ids = sorted(list(set(bus2subs.values())))

    country = 'US'
    substations = pd.DataFrame.from_records(
        index='id',
        data={'id': substation_ids,
              'country': [country] * len(substation_ids),
              },
    )

    # A substation can have multiple voltage levels In out case, one Bus has one
    # voltage level
    voltage_levels = pd.DataFrame.from_records(
        index='id',
        data={"id": ["VL{}".format(item) for item in ss.Bus.idx.v],
              "substation_id": [vl2subs[bus] for bus in ss.Bus.idx.v],
              "topology_kind": "BUS_BREAKER",
              "nominal_v": ss.Bus.Vn.v,
              }
    )

    return substations, voltage_levels


def _make_buses(ss):
    buses = pd.DataFrame.from_records(index='id', data={
        'voltage_level_id': ["VL{}".format(item) for item in ss.Bus.idx.v],
        'id':  ["{}".format(item) for item in ss.Bus.idx.v],
    })

    return buses


def _make_tline_tf(ss):
    """
    Generate line and 2-winding transformer data.

    """

    tf_idx = ss.Line.get_tf_idx()
    tline_idx = ss.Line.get_tline_idx()

    tline_pos = ss.Line.idx2uid(tline_idx)
    tf_pos = ss.Line.idx2uid(tf_idx)

    transformers = pd.DataFrame.from_records(index='id', data={
        'id': ['{}'.format(i) for i in ss.Line.get("idx", tf_idx)],
        'name': ['{}'.format(i) for i in ss.Line.get("name", tf_idx)],
        'voltage_level1_id': ["VL{}".format(item) for item in ss.Line.get("bus1", tf_idx)],
        'voltage_level2_id': ["VL{}".format(item) for item in ss.Line.get("bus2", tf_idx)],
        'bus1_id': ["{}".format(item) for item in ss.Line.get("bus1", tf_idx)],
        'connectable_bus1_id': ["{}".format(item) for item in ss.Line.get("bus1", tf_idx)],
        'bus2_id': ["{}".format(item) for item in ss.Line.get("bus2", tf_idx)],
        'connectable_bus2_id': ["{}".format(item) for item in ss.Line.get("bus2", tf_idx)],
        'rated_u1': ss.Bus.get("Vn", ss.Line.get("bus1", tf_idx)),
        'rated_u2': ss.Bus.get("Vn", ss.Line.get("bus2", tf_idx)),
        'rated_s': 9999 * np.ones(len(tf_idx)),
        'r': ss.Line.get("r", tf_idx) * ss.Line.bases['Zb'][tf_pos],
        'x': ss.Line.get("x", tf_idx) * ss.Line.bases['Zb'][tf_pos],
        'g': ss.Line.get("g", tf_idx) / ss.Line.bases['Zb'][tf_pos],
        'b': ss.Line.get("b", tf_idx) / ss.Line.bases['Zb'][tf_pos],
    })

    lines = pd.DataFrame.from_records(index='id', data={
        'id': ['{}'.format(i) for i in ss.Line.get("idx", tline_idx)],
        'name': ['{}'.format(i) for i in ss.Line.get("name", tline_idx)],
        'voltage_level1_id': ["VL{}".format(item) for item in ss.Line.get("bus1", tline_idx)],
        'voltage_level2_id': ["VL{}".format(item) for item in ss.Line.get("bus2", tline_idx)],
        'bus1_id': ["{}".format(item) for item in ss.Line.get("bus1", tline_idx)],
        'bus2_id': ["{}".format(item) for item in ss.Line.get("bus2", tline_idx)],
        'r': ss.Line.get("r", tline_idx) * ss.Line.bases['Zb'][tline_pos],
        'x': ss.Line.get("x", tline_idx) * ss.Line.bases['Zb'][tline_pos],
        'g1': ss.Line.get("g1", tline_idx) / ss.Line.bases['Zb'][tline_pos],
        'b1': ss.Line.get("b1", tline_idx) / ss.Line.bases['Zb'][tline_pos],
        'g2': ss.Line.get("g2", tline_idx) / ss.Line.bases['Zb'][tline_pos],
        'b2': ss.Line.get("b2", tline_idx) / ss.Line.bases['Zb'][tline_pos],
    })

    return lines, transformers


def _make_loads(ss):

    loads = pd.DataFrame.from_records(index='id', data={
        'voltage_level_id': ["VL{}".format(item) for item in ss.PQ.bus.v],
        'id': ["{}".format(item) for item in ss.PQ.idx.v],
        'bus_id': ["{}".format(item) for item in ss.PQ.bus.v],
        'p0': ss.PQ.p0.v * ss.config.mva,
        'q0': ss.PQ.q0.v * ss.config.mva,
    })

    return loads


def _make_generators(ss):
    generators = pd.DataFrame.from_records(index='id', data={
        'voltage_level_id': [f"VL{item}" for item in ss.PV.bus.v] + [f"VL{item}" for item in ss.Slack.bus.v],
        'id': [f"GEN{item}" for item in ss.PV.idx.v] + [f"GEN{item}" for item in ss.Slack.idx.v],
        'bus_id': [f"{item}" for item in ss.PV.bus.v] + [f"{item}" for item in ss.Slack.bus.v],
        'target_p': np.hstack([ss.PV.p0.v, ss.Slack.p0.v]) * ss.config.mva,
        'min_p': np.hstack([ss.PV.pmin.v, ss.Slack.pmin.v]) * ss.config.mva,
        'max_p': np.hstack([ss.PV.pmax.v, ss.Slack.pmax.v]) * ss.config.mva,
        'target_v': np.hstack([ss.PV.v0.v * ss.PV.Vn.v, ss.Slack.v0.v * ss.Slack.Vn.v]),
        'voltage_regulator_on': True
    })

    return generators


def _make_shunts(ss):
    shunt_df = pd.DataFrame.from_records(index='id', data={
        'id': [f"SHN{item}" for item in ss.Shunt.idx.v],
        'name': [f"{item}" for item in ss.Shunt.name.v],
        'model_type': 'NON_LINEAR',
        'section_count': 1,
        'target_v': ss.Shunt.Vn.v,
        'target_deadband': 0,
        'voltage_level_id': [f'VL{item}' for item in ss.Shunt.bus.v],
        'bus_id': [f'{item}' for item in ss.Shunt.bus.v],
    })

    shunt_model_df = pd.DataFrame.from_records(
        index='id',
        data={"id": [f"SHN{item}" for item in ss.Shunt.idx.v],
              "g": ss.Shunt.g.v / ss.Shunt.bases['Zb'],
              "b": ss.Shunt.b.v / ss.Shunt.bases['Zb'],
              }
    )

    return shunt_df, shunt_model_df
