"""
ZIP Load.
"""

from andes.core import (ModelData, IdxParam, NumParam, Model,
                        ConstService, ExtParam, ExtService, ExtAlgeb)
from andes.core.service import InitChecker


class ZIPData(ModelData):
    """
    Data for ZIP load initialized after power flow.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.pq = IdxParam(model='PQ', mandatory=True,
                           info='idx of the PQ to replace',
                           )

        self.kpp = NumParam(info='Percentage of active power',
                            mandatory=True,
                            tex_name='K_{pp}',
                            )
        self.kpi = NumParam(info='Percentage of active current',
                            mandatory=True,
                            tex_name='K_{pi}',
                            )
        self.kpz = NumParam(info='Percentage of conductance',
                            mandatory=True,
                            tex_name='K_{pz}',
                            )

        self.kqp = NumParam(info='Percentage of reactive power',
                            mandatory=True,
                            tex_name='K_{qp}',
                            )
        self.kqi = NumParam(info='Percentage of reactive current',
                            mandatory=True,
                            tex_name='K_{qi}',
                            )
        self.kqz = NumParam(info='Percentage of susceptance',
                            mandatory=True,
                            tex_name='K_{qz}',
                            )


class ZIPModel(Model):
    """
    Model for ZIP load.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'DynLoad'
        self.flags.tds = True

        self.kps = ConstService(v_str='kpp + kpi + kpz',
                                tex_name='K_{psum}',
                                )
        self.kqs = ConstService(v_str='kqp + kqi + kqz',
                                tex_name='K_{qsum}', )
        self.kpc = InitChecker(u=self.kps, equal=100.0,
                               tex_name='K_{pc}',
                               info='total `kp` and 100',
                               )
        self.kqc = InitChecker(u=self.kqs, equal=100.0,
                               tex_name='K_{qc}',
                               info='total `kq` and 100',
                               )

        # convert percentages to decimals
        self.rpp = ConstService(v_str='u * kpp / 100',
                                tex_name='r_{pp}',
                                )
        self.rpi = ConstService(v_str='u * kpi / 100',
                                tex_name='r_{pi}',
                                )
        self.rpz = ConstService(v_str='u * kpz / 100',
                                tex_name='r_{pz}',
                                )

        self.rqp = ConstService(v_str='u * kqp / 100',
                                tex_name='r_{qp}',
                                )
        self.rqi = ConstService(v_str='u * kqi / 100',
                                tex_name='r_{qi}',
                                )
        self.rqz = ConstService(v_str='u * kqz / 100',
                                tex_name='r_{qz}',
                                )

        self.bus = ExtParam(model='PQ', src='bus', indexer=self.pq,
                            info='retrieved bux idx',
                            export=False,
                            )

        self.p0 = ExtService(model='PQ', src='Ppf', indexer=self.pq,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='PQ', src='Qpf', indexer=self.pq,
                             tex_name='Q_0',
                             )
        self.v0 = ExtService(model='Bus', src='v', indexer=self.bus,
                             tex_name='V_0',
                             )

        # calculate initial powers, equivalent current, and equivalent z
        self.pp0 = ConstService(v_str='p0 * rpp',
                                tex_name='P_{p0}',
                                )
        self.pi0 = ConstService(v_str='p0 * rpi / v0',
                                tex_name='P_{i0}',
                                )
        self.pz0 = ConstService(v_str='p0 * rpz / v0 / v0',
                                tex_name='P_{z0}',
                                )

        self.qp0 = ConstService(v_str='q0 * rqp',
                                tex_name='Q_{p0}',
                                )
        self.qi0 = ConstService(v_str='q0 * rqi / v0',
                                tex_name='Q_{i0}',
                                )
        self.qz0 = ConstService(v_str='q0 * rqz / v0 / v0',
                                tex_name='Q_{z0}',
                                )

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus,
                          tex_name=r'\theta',
                          e_str='pp0 + pi0*v + pz0*v*v',
                          ename='P',
                          tex_ename='P',
                          )

        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus,
                          tex_name='V',
                          e_str='qp0 + qi0*v + qz0*v*v',
                          ename='Q',
                          tex_ename='Q',
                          )

    def v_numeric(self, **kwargs):
        """
        Disable the linked PQs.
        """
        self.system.groups['StaticLoad'].set(src='u', idx=self.pq.v, attr='v', value=0)


class ZIP(ZIPData, ZIPModel):
    """
    ZIP load model (polynomial load).
    This model is initialized after power flow.

    Please check the config of PQ to avoid double counting.
    If this ZIP model is in use, one should typically set
    `p2p=1.0` and `q2q=1.0` while leaving the others
    (`p2i`, `p2z`, `q2i`, `q2z`, and `pq2z`) as zeros.
    This setting allows one to impose the desired powers
    by the static PQ and to convert them based on the percentage
    specified in the ZIP.

    The percentages for active power, (`kpp`, `kpi`, and `kpz`)
    must sum up to 100. Otherwise, initialization will fail.
    The same applies to the reactive power percentages.
    """

    def __init__(self, system, config):
        ZIPData.__init__(self)
        ZIPModel.__init__(self, system, config)
