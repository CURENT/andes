"""
AC transmission line in dynamic phasor (shift frequency) formulation.
"""


from andes.core import (ModelData, IdxParam,
                        Model, ExtAlgeb, ConstService, ExtParam, State)


class DPLineData(ModelData):
    """
    Data for Line.
    """

    def __init__(self):
        super().__init__()

        self.line = IdxParam(info='Line index', model='Line')


class DPLine(DPLineData, Model):
    """
    WIP

    AC transmission line model in dynamic phasor (shift frequency) formulation.

    The connectivity `u` logic is as follows:
    - `u = 0` means the line is not replaced by DPLine. The line remains in the
      phasor model, and the `u` of the phasor Line device applies.
    - `u = 1` means the line is replaced by DPLine. Still, the connectivity
      status of the DP line depends on the `u` of the phasor Line device.

    """

    def __init__(self, system=None, config=None):
        DPLineData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACLine'
        self.flags.pflow = False
        self.flags.tds = True
        self.flags.tds_init = True

        self.bus1 = ExtParam(model='Line', src='bus1', indexer=self.line, export=False)
        self.bus2 = ExtParam(model='Line', src='bus2', indexer=self.line, export=False)

        self.a1 = ExtAlgeb(model='Bus', src='a', indexer=self.bus1, tex_name='a_1',
                           info='phase angle of the from bus',
                           ename='Pij',
                           tex_ename='P_{ij}',
                           )
        self.a2 = ExtAlgeb(model='Bus', src='a', indexer=self.bus2, tex_name='a_2',
                           info='phase angle of the to bus',
                           ename='Pji',
                           tex_ename='P_{ji}',
                           )
        self.v1 = ExtAlgeb(model='Bus', src='v', indexer=self.bus1, tex_name='v_1',
                           info='voltage magnitude of the from bus',
                           ename='Qij',
                           tex_ename='Q_{ij}',
                           )
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.bus2, tex_name='v_2',
                           info='voltage magnitude of the to bus',
                           ename='Qji',
                           tex_ename='Q_{ji}',
                           )

        self.ul = ExtParam(model='Line', src='u', indexer=self.line, tex_name='u', export=False)
        self.g1 = ExtParam(model='Line', src='g1', indexer=self.line, tex_name='g_1', export=False)
        self.g2 = ExtParam(model='Line', src='g2', indexer=self.line, tex_name='g_2', export=False)
        self.b1 = ExtParam(model='Line', src='b1', indexer=self.line, tex_name='b_1', export=False)
        self.b2 = ExtParam(model='Line', src='b2', indexer=self.line, tex_name='b_2', export=False)
        self.r = ExtParam(model='Line', src='r', indexer=self.line, tex_name='r', export=False)
        self.x = ExtParam(model='Line', src='x', indexer=self.line, tex_name='x', export=False)
        self.g = ExtParam(model='Line', src='g', indexer=self.line, tex_name='g', export=False)
        self.b = ExtParam(model='Line', src='b', indexer=self.line, tex_name='b', export=False)
        self.tap = ExtParam(model='Line', src='tap', indexer=self.line, tex_name='tap', export=False)

        self.gh = ConstService(tex_name='g_h')
        self.bh = ConstService(tex_name='b_h')
        self.gk = ConstService(tex_name='g_k')
        self.bk = ConstService(tex_name='b_k')

        self.yh = ConstService(tex_name='y_h', vtype=complex)
        self.yk = ConstService(tex_name='y_k', vtype=complex)
        self.yhk = ConstService(tex_name='y_{hk}', vtype=complex)

        self.ghk = ConstService(tex_name='g_{hk}')
        self.bhk = ConstService(tex_name='b_{hk}')

        self.itap = ConstService(tex_name='1/t_{ap}')
        self.itap2 = ConstService(tex_name='1/t_{ap}^2')

        self.Leq = ConstService(v_str='x/(2*pi*60)')

        self.ue = ConstService(v_str='ul * u')

        self.gh.v_str = 'g1 + 0.5 * g'
        self.bh.v_str = 'b1 + 0.5 * b'
        self.gk.v_str = 'g2 + 0.5 * g'
        self.bk.v_str = 'b2 + 0.5 * b'

        self.yh.v_str = 'u * (gh + 1j * bh)'
        self.yk.v_str = 'u * (gk + 1j * bk)'
        self.yhk.v_str = 'u/((r+1e-8) + 1j*(x+1e-8))'

        self.ghk.v_str = 're(yhk)'
        self.bhk.v_str = 'im(yhk)'

        self.itap.v_str = '1/tap'
        self.itap2.v_str = '1/tap/tap'

        self.r2x2 = ConstService(v_str='r*r + x*x')

        # in implicit form but has the same dq-axis alignment as in ANDES implementation)
        self.idd = State(info='real current',
                         tex_name='idd',
                         v_str='ue * (r*v1*sin(a1)/r2x2 - r*v2*sin(a2)/r2x2 - v1*x*cos(a1)/r2x2 +'
                               '      v2*x*cos(a2)/r2x2)',
                         e_str='ue * (-x*iqq - r*idd - v2*sin(a2) + v1*sin(a1)) + (1-ue) * idd',
                         t_const=self.Leq,
                         )

        self.iqq = State(info='real current',
                         tex_name='iqq',
                         v_str='ue * (r*v1*cos(a1)/r2x2 - r*v2*cos(a2)/r2x2 + v1*x*sin(a1)/r2x2 -'
                               '      v2*x*sin(a2)/r2x2)',
                         e_str='ue * (x*idd - r*iqq - v2*cos(a2) + v1*cos(a1)) + (1-ue) * iqq',
                         t_const=self.Leq,
                         )

        self.a1.e_str = 'ue * (idd*v1*sin(a1) + iqq*v1*cos(a1))'

        self.v1.e_str = 'ue * (-idd*v1*cos(a1) + iqq*v1*sin(a1))'

        self.a2.e_str = 'ue * (-idd*v2*sin(a2) - iqq*v2*cos(a2))'

        self.v2.e_str = 'ue * (idd*v2*cos(a2) - iqq*v2*sin(a2))'

    def v_numeric(self, **kwargs):
        """
        Disable phasor lines that have been replaced by DP lines.
        """

        mask_idx = [self.line.v[i] for i in range(self.n) if self.u.v[i] == 1]
        self.system.groups['ACLine'].set(src='u', idx=mask_idx, attr='v', value=0)
