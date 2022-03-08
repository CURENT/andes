"""
Simple pandapower (2.7.0) interface
"""
import pandapower as pp
from math import pi
from numpy import sign
import pandas as pd


def to_pandapower(ssa):
    """
    Convert ADNES system (ssa) to pandapower network (ssp).

    The power flow of `ssp` is consistent with ANDES.

    Line limts are set as 99999.0 in `ssp`.

    Generator cost is not included in the conversion.

    `SynGen` equipped with `Exciter` in `ssa` is considered as `controllable=True` in `ssp.gen`.
    """
    # create PP net
    ssp = pp.create_empty_network(f_hz=ssa.config.freq,
                                  sn_mva=ssa.config.mva,
                                  )

    # 1) bus
    ssa_bus = ssa.Bus.as_df().copy()
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

    # 2) line
    # TODO: 1) from- and to- sides `Y`; 2)`g`
    ssa_mva = ssp.sn_mva
    omega = pi * ssp.f_hz

    ssa_line = ssa.Line.as_df().merge(ssa.Bus.as_df()[['idx', 'Vn']].rename(
        columns={"idx": "bus1", "Vn": "Vb"}), on='bus1', how='left')

    ssa_line['Zb'] = ssa_line["Vb"]**2 / ssa_line["Sn"]
    ssa_line['R'] = ssa_line["r"] * ssa_line['Zb']  # ohm
    ssa_line['X'] = ssa_line["x"] * ssa_line['Zb']  # ohm
    ssa_line['C'] = ssa_line["b"] / ssa_line['Zb'] / omega * 1e9 / 2  # nF
    # ssa_line['G'] = ssa_line["g"] * ssa_line['Yb'] * 1e6  # mS

    # the line limits are set to be large
    ll_ka = len(ssa_line) * [99999]

    # line index
    ssl = ssa_line.copy()
    ssl['uidx'] = ssl.index
    index_line = ssl['uidx'][ssl['Vn1'] == ssl['Vn2']][ssl['trans'] == 0]
    # 2a) line
    for num, uid in enumerate(index_line):
        from_bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_line["bus1"].iloc[uid]].values[0]
        to_bus_name = ssa_bus["name"][ssa_bus["idx"] == ssa_line["bus2"].iloc[uid]].values[0]
        from_bus = pp.get_element_index(ssp, 'bus', name=from_bus_name)
        to_bus = pp.get_element_index(ssp, 'bus', name=to_bus_name)

        pp.create_line_from_parameters(net=ssp,
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
    # 2b) transformer
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
        sn = 99999.0  # ssa_line['Sn'].iloc[uid]
        baseMVA = ssa_mva

        ratio_1 = (ssa_line['tap'].iloc[uid] - 1) * 100
        i0_percent = -ssa_line['b'].iloc[uid] * 100 * baseMVA / sn

        pp.create_transformer_from_parameters(net=ssp,
                                              hv_bus=hv_bus,
                                              lv_bus=lv_bus,
                                              sn_mva=sn,
                                              vn_hv_kv=vn_hv_kv,
                                              vn_lv_kv=vn_lv_kv,
                                              vk_percent=sign(xk) * zk * sn * 100 / baseMVA,
                                              vkr_percent=rk * sn * 100 / baseMVA,
                                              max_loading_percent=100,
                                              pfe_kw=0, i0_percent=i0_percent,
                                              shift_degree=ssa_line['phi'].iloc[uid]*180/pi,
                                              tap_step_percent=abs(ratio_1), tap_pos=sign(ratio_1),
                                              tap_side=tap_side, tap_neutral=0,
                                              index=num)

    # 3) load
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

    # build Exciter df
    exc_cols = ['idx', 'u', 'name', 'syn']
    ssa_exc = pd.DataFrame(columns=exc_cols)
    for key in ssa.Exciter.models:
        exc = getattr(ssa, key)
        ssa_exc = pd.concat([ssa_exc, exc.as_df()[exc_cols]], axis=0)

    # Consider the SynGen equipped with Exciter as controllable
    ssa_sg_ctr = pd.merge(ssa_syg.rename(columns={"idx": "syn"}), ssa_exc[["syn", "idx"]], how="left", on="syn")
    ssa_sg_out = ssa_sg_ctr.rename(columns={"bus": "bus_idx"})
    ssa_sg_out["ctrl"] = bool(str(ssa_sg_ctr[["idx"]]))
    ssa_sg = ssa_sg.merge(ssa_sg_out[["bus_idx", "ctrl"]], on="bus_idx", how="left")

    # assign slack bus
    ssa_sg["slack"] = False
    ssa_sg["slack"][ssa_sg["bus_idx"] == ssa.Slack.bus.v[0]] = True

    # compute the actual value
    calc_cols = ['p0', 'q0', 'pmax', 'pmin', 'qmax', 'qmin']
    for col in calc_cols:
        ssa_sg[col] = ssa_sg[col] * ssa_sg["Sn"]

    # conversion
    # a) `PV` with negative `p0` -> load
    # b) `Slack` -> gen
    # c) `PV` with non-negative `p0`-> gen
    for uid in ssa_sg.index:
        if ssa_sg["p0"].iloc[uid] < 0:
            pp.create_load(net=ssp,
                           name="PV"+str(ssa_sg["name"].iloc[uid]),
                           bus=ssa_sg["pp_id"].iloc[uid],
                           sn_mva=ssa_sg["Sn"].iloc[uid],
                           p_mw=-1*ssa_sg["p0"].iloc[uid],
                           q_mvar=-1*ssa_sg["q0"].iloc[uid],
                           in_service=ssa_sg["u"].iloc[uid],
                           controllable=False,
                           #    index=uid,
                           )
        else:
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
    return ssp
