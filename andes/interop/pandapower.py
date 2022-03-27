"""
Simple pandapower (2.7.0) interface
"""

import logging
import numpy as np

from numpy import pi
from andes.shared import pd, rad2deg, deg2rad

try:
    import pandapower as pp
except ImportError:
    pp = None


logger = logging.getLogger(__name__)


def make_link_table(ssa):
    """
    Build the link table for generators and generator controllers in an ADNES
    System.

    Parameters
    ----------
    ssa :
        The ADNES system to link

    Returns
    -------
    DataFrame

        Each column in the output Dataframe contains the ``idx`` of linked
        ``StaticGen``, ``Bus``, ``SynGen``, ``Exciter``, and ``TurbineGov``.
    """
    # build StaticGen df
    sg_cols = ['name', 'idx', 'bus']
    ssa_sg = pd.DataFrame(columns=sg_cols)
    for key in ssa.StaticGen.models:
        sg = getattr(ssa, key)
        ssa_sg = pd.concat([ssa_sg, sg.as_df()[sg_cols]], axis=0)
    # build TurbineGov df
    gov_cols = ['idx', 'syn']
    ssa_gov = pd.DataFrame(columns=gov_cols)
    for key in ssa.TurbineGov.models:
        gov = getattr(ssa, key)
        ssa_gov = pd.concat([ssa_gov, gov.as_df()[gov_cols]], axis=0)
    # build Exciter df
    exc_cols = ['idx', 'syn']
    ssa_exc = pd.DataFrame(columns=exc_cols)
    for key in ssa.Exciter.models:
        exc = getattr(ssa, key)
        ssa_exc = pd.concat([ssa_exc, exc.as_df()[exc_cols]], axis=0)
    # build SynGen df
    syn_cols = ['idx', 'bus']
    ssa_syn = pd.DataFrame(columns=syn_cols)
    for key in ssa.SynGen.models:
        syn = getattr(ssa, key)
        ssa_syn = pd.concat([ssa_syn, syn.as_df()[syn_cols]], axis=0)

    # output
    ssa_bus = ssa.Bus.as_df()[['name', 'idx']]
    ssa_key = pd.merge(left=ssa_sg.rename(columns={'name': 'stg_name', 'idx': 'stg_idx', 'bus': 'bus_idx'}),
                       right=ssa_bus.rename(columns={'name': 'bus_name', 'idx': 'bus_idx'}),
                       how='left',
                       on='bus_idx')
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_syn.rename(columns={'idx': 'syn_idx', 'bus': 'bus_idx'}),
                       how='left',
                       on='bus_idx')
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_exc.rename(columns={'idx': 'exc_idx', 'syn': 'syn_idx'}),
                       how='left',
                       on='syn_idx')
    ssa_key = pd.merge(left=ssa_key,
                       right=ssa_gov.rename(columns={'idx': 'gov_idx', 'syn': 'syn_idx'}),
                       how='left',
                       on='syn_idx')
    cols = ['stg_name', 'stg_idx', 'bus_idx', 'syn_idx', 'exc_idx', 'gov_idx', 'bus_name']
    return ssa_key[cols]


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
    The pandapower net and the ANDES system must have same base MVA.
    """

    pp.runopp(ssp, **kwargs)
    # take dispatch results from pp
    ssp_res = pd.concat([ssp.gen['name'], ssp.res_gen[['p_mw', 'q_mvar', 'vm_pu']]], axis=1)

    ssp_res = pd.merge(left=ssp_res,
                       right=ssp.gen[['name', 'bus', 'controllable']],
                       how='left', on='name')
    ssp_res = pd.merge(left=ssp_res,
                       right=ssp.bus[['name']].reset_index().rename(
                           columns={'index': 'bus', 'name': 'bus_name'}),
                       how='left', on='bus')
    ssp_res = pd.merge(left=ssp_res,
                       right=link_table[['bus_name', 'bus_idx', 'syn_idx', 'gov_idx', 'stg_idx', 'exc_idx']],
                       how='left', on='bus_name')
    ssp_res['p'] = ssp_res['p_mw'] / ssp.sn_mva
    ssp_res['q'] = ssp_res['q_mvar'] / ssp.sn_mva
    col = ['name', 'p', 'q', 'vm_pu', 'bus_name', 'bus_idx',
           'controllable', 'syn_idx', 'gov_idx', 'exc_idx', 'stg_idx']
    return ssp_res[col]


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
    # check dim
    if gen_cost.shape[0] != ssp.gen[ssp.gen['controllable']].shape[0]:
        print('Input cost function size does not match controllable gen size.')

    for num, uid in enumerate(ssp.gen[ssp.gen['controllable']].index):
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


def to_pandapower(ssa, ctrl=[], verify=True):
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

      - Line limts are set as 99999.0 in the output network.
      - Generator cost is not included in the conversion. Use ``add_gencost()``
        to add cost data.
      - By default, ``SynGen`` equipped with ``TurbineGov`` in the ANDES System is converted
        to generators with ``controllable=True`` in pp's network.
      - By default, ``SynGen`` that has no ``TurbineGov`` and ``DG`` in the ANDES System
        is converted to generators with ``controllable=False`` in pp's network.
    """
    if pp is None:
        raise ImportError("Please install pandapower to continue")

    # create a PP network
    ssp = pp.create_empty_network(f_hz=ssa.config.freq,
                                  sn_mva=ssa.config.mva,
                                  )

    # --- 1. convert buses ---
    ssa_bus = ssa.Bus.as_df()
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

    # --- 2. convert Line ---
    # TODO: 1) from- and to- sides `Y`; 2)`g`
    ssa_mva = ssp.sn_mva
    omega = 2 * pi * ssp.f_hz

    ssa_bus_slice = ssa.Bus.as_df()[['idx', 'Vn']].rename(columns={"idx": "bus1", "Vn": "Vb"})
    ssa_line = ssa.Line.as_df().merge(ssa_bus_slice, on='bus1', how='left')

    ssa_line['Zb'] = ssa_line["Vb"]**2 / ssa_line["Sn"]
    ssa_line['Yb'] = ssa_line["Sn"] / ssa_line["Vb"]**2
    ssa_line['R'] = ssa_line["r"] * ssa_line['Zb']  # ohm
    ssa_line['X'] = ssa_line["x"] * ssa_line['Zb']  # ohm
    ssa_line['C'] = ssa_line["b"] / ssa_line['Zb'] / omega * 1e9  # nF
    ssa_line['G'] = ssa_line["g"] * ssa_line['Yb'] * 1e6  # mS

    # find index for transmission lines (i.e., non-transformers)
    ssl = ssa_line
    ssl['uidx'] = ssl.index
    index_line = ssl['uidx'][ssl['Vn1'] == ssl['Vn2']][ssl['trans'] == 0]
    ll_ka = len(ssa_line) * [99999]  # set large line limits

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
            max_i_ka=ll_ka[uid],
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
        baseMVA = ssa_mva

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

    # --- 3. load ---
    ssa_pq = ssa.PQ.as_df().copy()
    ssa_pq['p_mw'] = ssa_pq["p0"] * ssa_mva
    ssa_pq['q_mvar'] = ssa_pq["q0"] * ssa_mva

    for uid in ssa_pq.index:
        bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_pq["bus"].iloc[uid]].values[0]
        bus = pp.get_element_index(ssp, 'bus', name=bus_name)
        pp.create_load(net=ssp,
                       name=ssa_pq["name"].iloc[uid],
                       bus=bus,
                       sn_mva=ssa_mva,
                       p_mw=ssa_pq['p_mw'].iloc[uid],
                       q_mvar=ssa_pq['q_mvar'].iloc[uid],
                       in_service=ssa_pq["u"].iloc[uid],
                       controllable=False,
                       index=uid,
                       type=None,
                       )

    # 4) shunt
    ssa_shunt = ssa.Shunt.as_df().copy()
    ssa_shunt['p_mw'] = ssa_shunt["g"] * ssa_mva
    ssa_shunt['q_mvar'] = ssa_shunt["b"] * (-1) * ssa_mva

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

    # 5) generator
    ssa_busn = ssa.Bus.as_df().copy().reset_index()[["uid", "idx"]].rename(
        columns={"uid": "pp_id", "idx": "bus_idx"})

    # build StaticGen df
    sg_cols = ['idx', 'u', 'name', 'Sn', 'bus', 'v0',
               'p0', 'q0', 'pmax', 'pmin', 'qmax', 'qmin',
               'vmax', 'vmin']
    ssa_sg = pd.DataFrame(columns=sg_cols)
    for key in ssa.StaticGen.models:
        sg = getattr(ssa, key)
        ssa_sg = pd.concat([ssa_sg, sg.as_df()[sg_cols]], axis=0)

    ssa_sg = pd.merge(ssa_sg.rename(columns={"bus": "bus_idx"}),
                      ssa_busn,
                      how="left", on="bus_idx")

    # build SynGen df
    syg_cols = ['idx', 'u', 'name', 'bus']
    ssa_syg = pd.DataFrame(columns=syg_cols)
    for key in ssa.SynGen.models:
        syg = getattr(ssa, key)
        ssa_syg = pd.concat([ssa_syg, syg.as_df()[syg_cols]], axis=0)

    # build TurbineGov df
    gov_cols = ['idx', 'u', 'name', 'syn']
    ssa_gov = pd.DataFrame(columns=gov_cols)
    for key in ssa.TurbineGov.models:
        gov = getattr(ssa, key)
        ssa_gov = pd.concat([ssa_gov, gov.as_df()[gov_cols]], axis=0)

    # By default, consider the StaticGen equipped with Exciter as controllable
    ssa_sg_ctr = ssa_sg[["bus_idx"]]
    if ctrl:
        if len(ctrl) != len(ssa_sg):
            raise ValueError("ctrl length does not match StaticGen length")
        ssa_sg_ctr["ctrl"] = [bool(x) for x in ctrl]
    else:
        ssa_sg_ctr["ctrl"] = [not x for x in make_link_table(ssa)["gov_idx"].isna()]
    ssa_sg = ssa_sg.merge(ssa_sg_ctr[["bus_idx", "ctrl"]], on="bus_idx", how="left")

    # assign slack bus
    ssa_sg["slack"] = False
    ssa_sg["slack"][ssa_sg["bus_idx"] == ssa.Slack.bus.v[0]] = True

    # compute the actual value
    calc_cols = ['p0', 'q0', 'pmax', 'pmin', 'qmax', 'qmin']
    for col in calc_cols:
        ssa_sg[col] = ssa_sg[col] * ssa_sg["Sn"]

    # fill the ctrl with False
    ssa_sg.fillna(value=False, inplace=True)

    # conversion
    for uid in ssa_sg.index:
        pp.create_gen(net=ssp,
                      slack=ssa_sg["slack"].iloc[uid],
                      bus=ssa_sg["pp_id"].iloc[uid],
                      p_mw=ssa_sg["p0"].iloc[uid],
                      vm_pu=ssa_sg["v0"].iloc[uid],
                      sn_mva=ssa_sg["Sn"].iloc[uid],
                      name=ssa_sg['name'].iloc[uid],
                      controllable=ssa_sg["ctrl"].iloc[uid],
                      in_service=ssa_sg["u"].iloc[uid],
                      max_p_mw=ssa_sg["pmax"].iloc[uid],
                      min_p_mw=ssa_sg["pmin"].iloc[uid],
                      max_q_mvar=ssa_sg["qmax"].iloc[uid],
                      min_q_mvar=ssa_sg["qmin"].iloc[uid],
                      index=uid,
                      )

    if verify:
        _verify_pf(ssa, ssp)

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
    pf_bus['v_diff'] = pf_bus['v_andes'] - pf_bus['v_pp']
    pf_bus['a_diff'] = pf_bus['a_andes'] - pf_bus['a_pp']

    if (np.max(np.abs(pf_bus['v_diff'])) < tol) and \
            (np.max(np.abs(pf_bus['a_diff'])) < tol):
        logger.info("Power flow results are consistent. Conversion is successful.")
        return True
    else:
        logger.warning("Warning: Power flow results are inconsistent. Pleaes check!")
        return False
