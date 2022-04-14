"""
Simple pandapower (2.7.0) interface
"""

import logging
import numpy as np
from functools import wraps

from numpy import pi
from andes.shared import pd, rad2deg, deg2rad
from andes.shared import pandapower as pp

logger = logging.getLogger(__name__)


def require_pandapower(f):
    """
    Decorator for functions that require pandapower.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            getattr(pp, '__version__')
        except AttributeError:
            raise ModuleNotFoundError("pandapower needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


def build_group_table(ssa, group, columns, mdl_name=[]):
    """
    Build the table for devices in a group in an ADNES System.

    Parameters
    ----------
    ssa : andes.system.System
        The ADNES system to build the table
    group : string
        The ADNES group
    columns : list of string
        The common columns of a group that to be included in the table.
    mdl_name : list of string
        The list of models that to be included in the table. Default as all models.

    Returns
    -------
    DataFrame

        The output Dataframe contains the columns from the device
    """
    group_df = pd.DataFrame(columns=columns)
    group = getattr(ssa, group)
    if not mdl_name:
        mdl_dict = getattr(group, 'models')
        for key in mdl_dict:
            mdl = getattr(ssa, key)
            group_df = pd.concat([group_df, mdl.as_df()[columns]], axis=0)
    else:
        for key in mdl_name:
            mdl = getattr(ssa, key)
            group_df = pd.concat([group_df, mdl.as_df()[columns]], axis=0)
    return group_df


def make_link_table(ssa):
    """
    Build the link table for generators and generator controllers in an ADNES
    System, including ``SynGen`` and ``DG`` for now.

    Parameters
    ----------
    ssa : andes.system.System
        The ADNES system to link

    Returns
    -------
    DataFrame

        Each column in the output Dataframe contains the ``idx`` of linked
        ``StaticGen``, ``Bus``, ``DG``, ``SynGen``, ``Exciter``, and ``TurbineGov``,
        ``gammap``, ``gammaq``.
    """
    # build StaticGen df
    ssa_stg = build_group_table(ssa, 'StaticGen', ['u', 'name', 'idx', 'bus'])
    # build TurbineGov df
    ssa_gov = build_group_table(ssa, 'TurbineGov', ['idx', 'syn'])
    # build Exciter df
    ssa_exc = build_group_table(ssa, 'Exciter', ['idx', 'syn'])
    # build SynGen df
    ssa_syg = build_group_table(ssa, 'SynGen', ['idx', 'bus', 'gen', 'gammap', 'gammaq'], ['GENCLS', 'GENROU'])
    # build DG df
    ssa_dg = build_group_table(ssa, 'DG', ['idx', 'bus', 'gen', 'gammap', 'gammaq'])

    # output
    ssa_bus = ssa.Bus.as_df()[['name', 'idx']]
    ssa_key = pd.merge(left=ssa_stg.rename(columns={'name': 'stg_name', 'idx': 'stg_idx',
                                                    'bus': 'bus_idx', 'u': 'stg_u'}),
                       right=ssa_bus.rename(columns={'name': 'bus_name', 'idx': 'bus_idx'}),
                       how='left', on='bus_idx')
    ssa_syg = pd.merge(left=ssa_key, how='right', on='stg_idx',
                       right=ssa_syg.rename(columns={'idx': 'syg_idx', 'gen': 'stg_idx'}))
    ssa_dg = pd.merge(left=ssa_key, how='right', on='stg_idx',
                      right=ssa_dg.rename(columns={'idx': 'dg_idx', 'gen': 'stg_idx'}))
    # TODO: Add RenGen
    ssa_key = pd.concat([ssa_syg, ssa_dg], axis=0)
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_exc.rename(columns={'idx': 'exc_idx', 'syn': 'syg_idx'}),
                       how='left', on='syg_idx')
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_gov.rename(columns={'idx': 'gov_idx', 'syn': 'syg_idx'}),
                       how='left', on='syg_idx')
    cols = ['stg_name', 'stg_u', 'stg_idx', 'bus_idx', 'dg_idx', 'syg_idx', 'exc_idx',
            'gov_idx', 'bus_name', 'gammap', 'gammaq']
    return ssa_key[cols]


@require_pandapower
def runopp_map(ssp, link_table, **kwargs):
    """
    Run OPF in pandapower using ``pp.runopp()`` and map results back to ANDES
    based on the link table.

    Parameters
    ----------
    ssp : pandapower network
        The pandapower network

    link_table : DataFrame
        The link table of ADNES system

    Returns
    -------
    DataFrame

        The DataFrame contains the OPF results with columns ``p_mw``,
        ``q_mvar``, ``vm_pu`` in p.u., and the corresponding ``idx`` of
        ``StaticGen``, ``Exciter``, ``TurbineGov`` in ANDES.

    Notes
    -----
      - The pandapower net and the ANDES system must have same base MVA.
      - Multiple ``DG`` connected to the same ``StaticGen`` will be converted to one generator.
        The power is dispatched to each ``DG`` by the power ratio ``gammap``
    """

    pp.runopp(ssp, **kwargs)

    # take dispatch results from pp
    ssp_gen = ssp.gen.rename(columns={'name': 'stg_idx'})
    ssp_res = pd.concat([ssp_gen[['stg_idx']],
                        ssp.res_gen[['p_mw', 'q_mvar', 'vm_pu']]], axis=1)

    ssp_res = pd.merge(left=ssp_res,
                       right=ssp_gen[['stg_idx', 'controllable']],
                       how='left', on='stg_idx')
    ssp_res = pd.merge(left=ssp_res, right=link_table,
                       how='left', on='stg_idx')
    ssp_res['p'] = ssp_res['p_mw'] * ssp_res['gammap'] / ssp.sn_mva
    ssp_res['q'] = ssp_res['q_mvar'] * ssp_res['gammaq'] / ssp.sn_mva
    col = ['stg_idx', 'p', 'q', 'vm_pu', 'bus_idx', 'controllable',
           'dg_idx', 'syg_idx', 'gov_idx', 'exc_idx']
    return ssp_res[col]


@require_pandapower
def add_gencost(ssp, gen_cost):
    """
    Add cost function to converted pandapower net `ssp`.

    The cost data follows the same format of pypower and matpower.

    Now only poly_cost is supported.

    Parameters
    ----------
    ssp :
        The pandapower net

    gen_cost : array
        generator cost data
    """

    # check dimension
    if gen_cost.shape[0] != ssp.gen.shape[0]:
        print('Input cost function size does not match gen size.')

    for num, uid in enumerate(ssp.gen.index):
        if gen_cost[num, 0] == 2:  # poly_cost
            pp.create_poly_cost(net=ssp,
                                element=uid,
                                et='gen',
                                cp2_eur_per_mw2=gen_cost[num, 4],
                                cp1_eur_per_mw=gen_cost[num, 5],
                                cp0_eur=gen_cost[num, 6])
        else:
            # TODO: piecewise linear
            continue

    return True


def _to_pp_bus(ssp, ssa_bus):
    """Create bus in pandapower net"""
    for uid in ssa_bus.index:
        pp.create_bus(net=ssp,
                      vn_kv=ssa_bus["Vn"].iloc[uid],
                      name=ssa_bus["name"].iloc[uid],
                      in_service=ssa_bus["u"].iloc[uid],
                      max_vm_pu=ssa_bus["vmax"].iloc[uid],
                      min_vm_pu=ssa_bus["vmin"].iloc[uid],
                      zone=ssa_bus["zone"].iloc[uid],
                      index=uid,
                      )
    return ssp


def _to_pp_line(ssa, ssp, ssa_bus):
    """Create line in pandapower net"""
    # TODO: 1) from- and to- sides `Y`; 2)`g`
    omega = 2 * pi * ssp.f_hz

    ssa_bus_slice = ssa.Bus.as_df()[['idx', 'Vn']].rename(columns={"idx": "bus1", "Vn": "Vb"})
    ssa_line = ssa.Line.as_df().merge(ssa_bus_slice, on='bus1', how='left')

    ssa_line['Zb'] = ssa_line["Vb"]**2 / ssa_line["Sn"]
    ssa_line['Yb'] = ssa_line["Sn"] / ssa_line["Vb"]**2
    ssa_line['R'] = ssa_line["r"] * ssa_line['Zb']  # ohm
    ssa_line['X'] = ssa_line["x"] * ssa_line['Zb']  # ohm
    ssa_line['C'] = ssa_line["b"] / ssa_line['Zb'] / omega * 1e9  # nF
    ssa_line['G'] = ssa_line["g"] * ssa_line['Yb'] * 1e6  # mS
    # default rate_a is 2000 MVA
    ssa_line['rate_a'] = ssa_line['rate_a'].replace(0, 2000)
    ssa_line['max_i_ka'] = ssa_line["rate_a"] / ssa_line['Vb']  # kA

    # find index for transmission lines (i.e., non-transformers)
    ssl = ssa_line
    ssl['uidx'] = ssl.index
    index_line = ssl['uidx'][ssl['Vn1'] == ssl['Vn2']][ssl['trans'] == 0]

    ssa_line['name'] = _rename(ssa_line['name'])
    # --- 2a. transmission lines ---
    for num, uid in enumerate(index_line):
        from_bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_line["bus1"].iloc[uid]].values[0]
        to_bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_line["bus2"].iloc[uid]].values[0]
        from_bus = pp.get_element_index(ssp, 'bus', name=from_bus_name)
        to_bus = pp.get_element_index(ssp, 'bus', name=to_bus_name)

        pp.create_line_from_parameters(
            net=ssp,
            name=ssa_line["name"].iloc[uid],
            from_bus=from_bus,
            to_bus=to_bus,
            in_service=ssa_line["u"].iloc[uid],
            length_km=1,
            r_ohm_per_km=ssa_line['R'].iloc[uid],
            x_ohm_per_km=ssa_line['X'].iloc[uid],
            c_nf_per_km=ssa_line['C'].iloc[uid],
            #    g_us_per_km = ssa_line['R'].iloc[uid],
            max_i_ka=ssa_line['max_i_ka'].iloc[uid],
            type='ol',
            max_loading_percent=100,
            index=num,
        )

    # --- 2b. transformer ---
    for num, uid in enumerate(ssa_line[~ssa_line.index.isin(index_line)].index):
        from_bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_line["bus1"].iloc[uid]].values[0]
        to_bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_line["bus2"].iloc[uid]].values[0]
        from_bus = pp.get_element_index(ssp, 'bus', name=from_bus_name)
        to_bus = pp.get_element_index(ssp, 'bus', name=to_bus_name)
        from_vn_kv = ssp.bus['vn_kv'].iloc[from_bus]
        to_vn_kv = ssp.bus['vn_kv'].iloc[to_bus]
        if from_vn_kv >= to_vn_kv:
            hv_bus = from_bus
            vn_hv_kv = from_vn_kv
            lv_bus = to_bus
            vn_lv_kv = to_vn_kv
            tap_side = 'hv'
        else:
            hv_bus = to_bus
            vn_hv_kv = to_vn_kv
            lv_bus = from_bus
            vn_lv_kv = from_vn_kv
            tap_side = 'lv'

        rk = ssa_line['r'].iloc[uid]
        xk = ssa_line['x'].iloc[uid]
        zk = (rk ** 2 + xk ** 2) ** 0.5
        sn = 99999.0
        baseMVA = ssp.sn_mva

        ratio_1 = (ssa_line['tap'].iloc[uid] - 1) * 100
        i0_percent = -ssa_line['b'].iloc[uid] * 100 * baseMVA / sn

        pp.create_transformer_from_parameters(
            net=ssp,
            hv_bus=hv_bus,
            lv_bus=lv_bus,
            sn_mva=sn,
            vn_hv_kv=vn_hv_kv,
            vn_lv_kv=vn_lv_kv,
            vk_percent=np.sign(xk) * zk * sn * 100 / baseMVA,
            vkr_percent=rk * sn * 100 / baseMVA,
            max_loading_percent=100,
            pfe_kw=0, i0_percent=i0_percent,
            shift_degree=ssa_line['phi'].iloc[uid] * rad2deg,
            tap_step_percent=abs(ratio_1), tap_pos=np.sign(ratio_1),
            tap_side=tap_side, tap_neutral=0,
            index=num)
    return ssp


def _to_pp_load(ssa, ssp, ssa_bus):
    """Create load in pandapower net"""
    ssa_pq = ssa.PQ.as_df()
    ssa_pq['p_mw'] = ssa_pq["p0"] * ssp.sn_mva
    ssa_pq['q_mvar'] = ssa_pq["q0"] * ssp.sn_mva

    ssa_pq['name'] = _rename(ssa_pq['name'])
    for uid in ssa_pq.index:
        bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_pq["bus"].iloc[uid]].values[0]
        bus = pp.get_element_index(ssp, 'bus', name=bus_name)
        pp.create_load(net=ssp,
                       name=ssa_pq["name"].iloc[uid],
                       bus=bus,
                       sn_mva=ssp.sn_mva,
                       p_mw=ssa_pq['p_mw'].iloc[uid],
                       q_mvar=ssa_pq['q_mvar'].iloc[uid],
                       in_service=ssa_pq["u"].iloc[uid],
                       controllable=False,
                       index=uid,
                       type=None,
                       )
    return ssp


def _to_pp_shunt(ssa, ssp, ssa_bus):
    """Create shunt in pandapower net"""
    ssa_shunt = ssa.Shunt.as_df()
    ssa_shunt['p_mw'] = ssa_shunt["g"] * ssp.sn_mva
    ssa_shunt['q_mvar'] = ssa_shunt["b"] * (-1) * ssp.sn_mva

    ssa_shunt['name'] = _rename(ssa_shunt['name'])
    for uid in ssa_shunt.index:
        bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_shunt["bus"].iloc[uid]].values[0]
        bus = pp.get_element_index(ssp, 'bus', name=bus_name)
        pp.create_shunt(net=ssp,
                        bus=bus,
                        p_mw=ssa_shunt['p_mw'].iloc[uid],
                        q_mvar=ssa_shunt['q_mvar'].iloc[uid],
                        vn_kv=ssa_shunt["Vn"].iloc[uid],
                        step=1,
                        max_step=1,
                        name=ssa_shunt["name"].iloc[uid],
                        in_service=ssa_shunt["u"].iloc[uid],
                        index=uid,
                        )
    return ssp


def _to_pp_gen_pre(ssa):
    """Create generator data in pandapower net"""
    # build StaticGen df
    stg_cols = ['idx', 'u', 'bus', 'v0', 'vmax', 'vmin']
    stg_calc_cols = ['p0', 'q0', 'pmax', 'pmin', 'qmax', 'qmin']
    ssa_stg = build_group_table(ssa, 'StaticGen', stg_cols + stg_calc_cols)
    ssa_stg['name'] = ssa_stg.idx
    ssa_stg.rename(inplace=True,
                   columns={"bus": "bus_idx",
                            "u": "stg_u",
                            "idx": "stg_idx"})
    # retrieve Bus idx in pp net
    ssa_busn = ssa.Bus.as_df().copy().reset_index()[["uid", "idx"]].rename(
        columns={"uid": "pp_bus", "idx": "bus_idx"})

    ssa_stg = pd.merge(ssa_stg, ssa_busn,
                       how="left", on="bus_idx")

    return ssa_stg


def _to_pp_gen(ssa, ssp, ctrl=[]):
    """Create shunt in pandapower net"""
    # TODO: Add RenGen
    ssa_sg = _to_pp_gen_pre(ssa)

    # assign slack bus
    ssa_sg["slack"] = False
    ssa_sg["slack"][ssa_sg["bus_idx"] == ssa.Slack.bus.v[0]] = True

    # compute the actual value
    stg_calc_cols = ['p0', 'q0', 'pmax', 'pmin', 'qmax', 'qmin']
    ssa_sg[stg_calc_cols] = ssa_sg[stg_calc_cols].apply(lambda x: x * ssp.sn_mva)
    if 'gammap' not in _to_pp_gen_pre(ssa).columns:
        ssa_sg['gammap'] = 1
        ssa_sg['gammaq'] = 1
    else:
        ssa_sg['p0'] = ssa_sg['p0'] * ssa_sg['gammap']
        ssa_sg['q0'] = ssa_sg['q0'] * ssa_sg['gammaq']

    # default controllable is determined by governor
    if len(ctrl) > 0:
        if len(ctrl) != len(ssa_sg):
            raise ValueError("ctrl length does not match StaticGen length")
    else:
        ctrl = [1] * len(ssa_sg)
    ssa_sg["ctrl"] = [bool(x) for x in ctrl]

    ssa_sg['name'] = _rename(ssa_sg['name'])
    # conversion
    for uid in ssa_sg.index:
        pp.create_gen(net=ssp,
                      slack=ssa_sg["slack"].iloc[uid],
                      bus=ssa_sg["pp_bus"].iloc[uid],
                      p_mw=ssa_sg["p0"].iloc[uid],
                      vm_pu=ssa_sg["v0"].iloc[uid],
                      sn_mva=ssp.sn_mva,
                      name=ssa_sg['name'].iloc[uid],
                      controllable=ssa_sg["ctrl"].iloc[uid],
                      in_service=ssa_sg["stg_u"].iloc[uid],
                      max_p_mw=ssa_sg["pmax"].iloc[uid],
                      min_p_mw=ssa_sg["pmin"].iloc[uid],
                      max_q_mvar=ssa_sg["qmax"].iloc[uid],
                      min_q_mvar=ssa_sg["qmin"].iloc[uid],
                      index=uid,
                      )
    return ssp


@require_pandapower
def to_pandapower(ssa, ctrl=[], verify=True, tol=1e-6):
    """
    Convert an ADNES system to a pandapower network for power flow studies.

    Parameters
    ----------
    ssa : andes.system.System
        The ADNES system to be converted
    ctrl : list
        The controlability of generators. The length should be the same with the
        number of ``StaticGen``.
        If not given, controllability of generators will be assigned by default.
        Example input: [1, 0, 1, ...]; ``PV`` first, then ``Slack``.

    Returns
    -------
    pandapower.auxiliary.pandapowerNet
        A pandapower net with the same bus, branch, gen, and load data as the
        ANDES system

    Notes
    -----
    Handling of the following parameters:

      - Generator cost is not included in the conversion. Use ``add_gencost()``
        to add cost data.
      - By default, all generators in ``ssp`` are controllable unless user-defined controllability
        is given
      - The online status of generators are determined by the online status of ``StaticGen``
        that connected to the ``SynGen`` or ``DG``
      - ``ssp.gen.name`` is from ``ssa.StaticGen.idx``, which should be unique
    """

    # create a PP network
    ssp = pp.create_empty_network(f_hz=ssa.config.freq,
                                  sn_mva=ssa.config.mva,
                                  )

    # build bus table
    ssa_bus = ssa.Bus.as_df()
    ssa_bus['name'] = _rename(ssa_bus['name'])

    # --- 1. convert buses ---
    ssp = _to_pp_bus(ssp, ssa_bus)

    # --- 2. convert Line ---
    ssp = _to_pp_line(ssa, ssp, ssa_bus)

    # --- 3. load ---
    ssp = _to_pp_load(ssa, ssp, ssa_bus)

    # --- 4. shunt ---
    ssp = _to_pp_shunt(ssa, ssp, ssa_bus)

    # --- 5. generator ---
    ssp = _to_pp_gen(ssa, ssp, ctrl)

    if verify:
        _verify_pf(ssa, ssp, tol)

    return ssp


def _verify_pf(ssa, ssp, tol=1e-6):
    """
    Verify power flow results.
    """
    ssa.PFlow.run()
    pp.runpp(ssp)

    pf_bus = ssa.Bus.as_df()[["name"]]

    # ssa
    pf_bus['v_andes'] = ssa.Bus.v.v
    pf_bus['a_andes'] = ssa.Bus.a.v

    # ssp
    pf_bus['v_pp'] = ssp.res_bus['vm_pu']
    pf_bus['a_pp'] = ssp.res_bus['va_degree'] * deg2rad

    # align ssa angle with slcka bus angle
    row_slack = np.argmin(np.abs(pf_bus['a_pp']))
    pf_bus['a_andes'] = pf_bus['a_andes'] - pf_bus['a_andes'].iloc[row_slack]

    pf_bus['v_diff'] = pf_bus['v_andes'] - pf_bus['v_pp']
    pf_bus['a_diff'] = pf_bus['a_andes'] - pf_bus['a_pp']

    if (np.max(np.abs(pf_bus['v_diff'])) < tol) and \
            (np.max(np.abs(pf_bus['a_diff'])) < tol):
        logger.info("Power flow results are consistent. Conversion is successful.")
        return True
    else:
        logger.warning("Warning: Power flow results are inconsistent. Pleaes check!")
        return False


def _rename(pds_in):
    """
    Rename the duplicated elelemts in a pandas.Series.
    """
    if pds_in.duplicated().any():
        pds = pds_in.copy()
        dp_index = pds.duplicated()[pds.duplicated()].index
        pds.iloc[dp_index] = pds.iloc[dp_index] + ' ' + pds.iloc[dp_index].index.astype(str)
        return pds
    else:
        return pds_in


# TODO: add test for make GSF
@require_pandapower
def make_GSF(ppn, verify=True, using_sparse_solver=False):
    """
    Build the Generation Shift Factor matrix of a pandapower net.

    Parameters
    ----------
    ppn : pandapower.network.Network
        Pandapower network
    verify : bool
        True to verify the GSF with that from DC power flow
    using_sparse_solver : bool
        True to use a sparse solver for pandapower maktPTDF

    Returns
    -------
    np.ndarray
        The GSF array
    """

    from pandapower.pypower.makePTDF import makePTDF
    from pandapower.pd2ppc import _pd2ppc

    # --- run DCPF ---
    pp.rundcpp(ppn)

    # --- compute PTDF ---
    _, ppci = _pd2ppc(ppn)
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"],
                    using_sparse_solver=using_sparse_solver)

    # --- get the gsf ---
    line_size = ppn.line.shape[0]
    gsf = ptdf[0:line_size, :]

    if verify:
        _verifyGSF(ppn, gsf)

    return gsf


def _verifyGSF(ppn, gsf, tol=1e-4):
    """Verify the GSF with DCPF"""
    # --- DCPF results ---
    rl = pd.concat([ppn.res_line['p_from_mw'], ppn.line[['from_bus', 'to_bus']]], axis=1)
    rp = _sumPF_ppn(ppn)
    rl_c = np.array(np.matrix(gsf) * np.matrix(rp.ngen).T)
    res_gap = rl.p_from_mw.values - rl_c.flatten()
    if np.abs(res_gap).max() <= tol:
        logger.info("GSF is consistent.")
    else:
        logger.warning("Warning: GSF is inconsistent. Pleaes check!")


def _sumPF_ppn(ppn):
    """Summarize PF results of a pandapower net"""
    rg = pd.concat([ppn.res_gen[['p_mw']], ppn.gen[['bus']]], axis=1).rename(columns={'p_mw': 'gen'})
    rd = pd.concat([ppn.res_load[['p_mw']], ppn.load[['bus']]], axis=1).rename(columns={'p_mw': 'demand'})
    rp = pd.DataFrame()
    rp['bus'] = ppn.bus.index
    rp = rp.merge(rg, on='bus', how='left')
    rp = rp.merge(rd, on='bus', how='left')
    rp.fillna(0, inplace=True)
    rp['ngen'] = rp['gen'] - rp['demand']
    rp = rp.groupby('bus').sum().reset_index(drop=True)
    rp['bus'] = rp.index
    return rp
