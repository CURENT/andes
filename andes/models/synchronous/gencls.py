"""
Classical generator implementation.
"""

import logging

from andes.core import ExtService, ConstService
from andes.models.synchronous.genbase import GENBaseData, GENBase, Flux0

logger = logging.getLogger(__name__)


class GENCLSModel:
    """
    Model for classical generator.
    """

    def __init__(self):
        # internal voltage and rotor angle calculation
        self.xq = ExtService(model='GENCLS', src='xd1', indexer=self.idx,
                             tex_name="{x'_d}",
                             )
        self._V = ConstService(v_str='v * exp(1j * a)',
                               tex_name='V_c',
                               vtype=complex,
                               )
        self._S = ConstService(v_str='p0 - 1j * q0',
                               tex_name='S',
                               vtype=complex,
                               )
        self._I = ConstService(v_str='_S / conj(_V)',
                               tex_name='I_c',
                               vtype=complex,
                               )
        self._E = ConstService(tex_name='E', vtype=complex)
        self._deltac = ConstService(tex_name=r'\delta_c', vtype=complex)
        self.delta0 = ConstService(tex_name=r'\delta_0')

        self.vdq = ConstService(v_str='u * (_V * exp(1j * 0.5 * pi - _deltac))',
                                tex_name='V_{dq}', vtype=complex)
        self.Idq = ConstService(v_str='u * (_I * exp(1j * 0.5 * pi - _deltac))',
                                tex_name='I_{dq}', vtype=complex)

        self.Id0 = ConstService(v_str='re(Idq)',
                                tex_name=r'I_{d0}')
        self.Iq0 = ConstService(v_str='im(Idq)',
                                tex_name=r'I_{q0}')
        self.vd0 = ConstService(v_str='re(vdq)',
                                tex_name=r'V_{d0}')
        self.vq0 = ConstService(v_str='im(vdq)',
                                tex_name=r'V_{q0}')

        self.tm0 = ConstService(tex_name=r'\tau_{m0}',
                                v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)')
        self.psid0 = ConstService(tex_name=r"\psi_{d0}",
                                  v_str='u * (ra * Iq0) + vq0')
        self.psiq0 = ConstService(tex_name=r"\psi_{q0}",
                                  v_str='-u * (ra * Id0) - vd0')
        self.vf0 = ConstService(tex_name=r'v_{f0}')

        # initialization of internal voltage and delta
        self._E.v_str = '_V + _I * (ra + 1j * xq)'
        self._deltac.v_str = 'log(_E / abs(_E))'
        self.delta0.v_str = 'u * im(_deltac)'

        self.Id.e_str += '+ xq * Id - vf'
        self.Iq.e_str += '+ xq * Iq'
        self.vf0.v_str = '(vq0 + ra * Iq0) + xq * Id0'


class GENCLS(GENBaseData, GENBase, GENCLSModel, Flux0):
    """
    Classical generator model.
    """

    def __init__(self, system, config):
        GENBaseData.__init__(self)
        GENBase.__init__(self, system, config)
        GENCLSModel.__init__(self)
        Flux0.__init__(self)
