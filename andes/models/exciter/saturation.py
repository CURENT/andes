from andes.core.common import dummify

from andes.core.block import Block
from andes.core.service import ConstService, FlagValue, InitChecker


class ExcExpSat(Block):
    r"""
    Exponential exciter saturation block to calculate
    A and B from E1, SE1, E2 and SE2.
    Input parameters will be corrected and the user will be warned.
    To disable saturation, set either E1 or E2 to 0.

    Parameters
    ----------
    E1 : BaseParam
        First point of excitation field voltage
    SE1: BaseParam
        Coefficient corresponding to E1
    E2 : BaseParam
        Second point of excitation field voltage
    SE2: BaseParam
        Coefficient corresponding to E2
    """

    def __init__(self, E1, SE1, E2, SE2, name=None, tex_name=None, info=None):
        Block.__init__(self, name=name, tex_name=tex_name, info=info)

        self._E1 = E1
        self._E2 = E2
        self._SE1 = SE1
        self._SE2 = SE2

        self.zE1 = FlagValue(self._E1, value=0.,
                             info='Flag non-zeros in E1',
                             tex_name='z^{E1}',
                             )
        self.zE2 = FlagValue(self._E2, value=0.,
                             info='Flag non-zeros in E2',
                             tex_name='z^{E2}',
                             )
        self.zSE1 = FlagValue(self._SE1, value=0.,
                              info='Flag non-zeros in SE1',
                              tex_name='z^{SE1}',
                              )
        self.zSE2 = FlagValue(self._SE2, value=0.,
                              info='Flag non-zeros in SE2',
                              tex_name='z^{SE2}')

        # disallow E1 = E2 != 0 since the curve fitting will fail
        self.E12c = InitChecker(
            self._E1, not_equal=self._E2,
            info='E1 and E2 after correction',
            error_out=True,
        )

        # data correction for E1, E2, SE1
        self.E1 = ConstService(
            tex_name='E^{1c}',
            info='Corrected E1 data',
        )
        self.E2 = ConstService(
            tex_name='E^{2c}',
            info='Corrected E2 data',
        )
        self.SE1 = ConstService(
            tex_name='SE^{1c}',
            info='Corrected SE1 data',
        )
        self.SE2 = ConstService(
            tex_name='SE^{2c}',
            info='Corrected SE2 data',
        )
        self.A = ConstService(info='Saturation gain',
                              tex_name='A^e',
                              )
        self.B = ConstService(info='Exponential coef. in saturation',
                              tex_name='B^e',
                              )
        self.vars = {
            'E1': self.E1,
            'E2': self.E2,
            'SE1': self.SE1,
            'SE2': self.SE2,
            'zE1': self.zE1,
            'zE2': self.zE2,
            'zSE1': self.zSE1,
            'zSE2': self.zSE2,
            'A': self.A,
            'B': self.B,
        }

    def define(self):
        r"""
        Notes
        -----
        The implementation solves for coefficients `A` and `B`
        which satisfy

        .. math ::
            E_1  S_{E1} = A e^{E1\times B}
            E_2  S_{E2} = A e^{E2\times B}

        The solutions are given by

        .. math ::
            E_{1} S_{E1} e^{ \frac{E_1 \log{ \left( \frac{E_2 S_{E2}} {E_1 S_{E1}} \right)} } {E_1 - E_2}}
            - \frac{\log{\left(\frac{E_2 S_{E2}}{E_1 S_{E1}} \right)}}{E_1 - E_2}
        """
        self.E1.v_str = f'{self._E1.name} + (1 - {self.name}_zE1)'
        self.E2.v_str = f'{self._E2.name} + 2*(1 - {self.name}_zE2)'

        self.SE1.v_str = f'{self._SE1.name} + (1 - {self.name}_zSE1)'
        self.SE2.v_str = f'{self._SE2.name} + 2*(1 - {self.name}_zSE2)'

        self.A.v_str = f'{self.name}_zE1*{self.name}_zE2 * ' \
                       f'{self.name}_E1*{self.name}_SE1*' \
                       f'exp({self.name}_E1*log({self.name}_E2*{self.name}_SE2/' \
                       f'({self.name}_E1*{self.name}_SE1))/({self.name}_E1-{self.name}_E2))'

        self.B.v_str = f'-log({self.name}_E2*{self.name}_SE2/({self.name}_E1*{self.name}_SE1))/' \
                       f'({self.name}_E1 - {self.name}_E2)'


class ExcQuadSat(Block):
    r"""
    Exponential exciter saturation block to calculate
    A and B from E1, SE1, E2 and SE2.
    Input parameters will be corrected and the user will be warned.
    To disable saturation, set either E1 or E2 to 0.

    Parameters
    ----------
    E1 : BaseParam
        First point of excitation field voltage
    SE1: BaseParam
        Coefficient corresponding to E1
    E2 : BaseParam
        Second point of excitation field voltage
    SE2: BaseParam
        Coefficient corresponding to E2
    """

    def __init__(self, E1, SE1, E2, SE2, name=None, tex_name=None, info=None):
        Block.__init__(self, name=name, tex_name=tex_name, info=info)

        self._E1 = dummify(E1)
        self._E2 = dummify(E2)
        self._SE1 = SE1
        self._SE2 = SE2

        self.zSE2 = FlagValue(self._SE2, value=0.,
                              info='Flag non-zeros in SE2',
                              tex_name='z^{SE2}')

        # data correction for E1, E2, SE1 (TODO)
        self.E1 = ConstService(
            tex_name='E^{1c}',
            info='Corrected E1 data',
        )
        self.E2 = ConstService(
            tex_name='E^{2c}',
            info='Corrected E2 data',
        )
        self.SE1 = ConstService(
            tex_name='SE^{1c}',
            info='Corrected SE1 data',
        )
        self.SE2 = ConstService(
            tex_name='SE^{2c}',
            info='Corrected SE2 data',
        )
        self.a = ConstService(info='Intermediate Sat coeff',
                              tex_name='a',
                              )
        self.A = ConstService(info='Saturation start',
                              tex_name='A^q',
                              )
        self.B = ConstService(info='Saturation gain',
                              tex_name='B^q',
                              )
        self.vars = {
            'E1': self.E1,
            'E2': self.E2,
            'SE1': self.SE1,
            'SE2': self.SE2,
            'zSE2': self.zSE2,
            'a': self.a,
            'A': self.A,
            'B': self.B,
        }

    def define(self):
        r"""
        Notes
        -----
        TODO.
        """

        self.E1.v_str = f'{self._E1.name}'
        self.E2.v_str = f'{self._E2.name}'
        self.SE1.v_str = f'{self._SE1.name}'
        self.SE2.v_str = f'{self._SE2.name} + 2 * (1 - {self.name}_zSE2)'

        self.a.v_str = f'(Indicator({self.name}_SE2>0)+Indicator({self.name}_SE2<0)) * ' \
                       f'sqrt({self.name}_SE1 * {self.name}_E1 /({self.name}_SE2 * {self.name}_E2))'

        self.A.v_str = f'{self.name}_E2 - ({self.name}_E1 - {self.name}_E2) / ({self.name}_a - 1)'

        self.B.v_str = f'(Indicator({self.name}_a>0)+Indicator({self.name}_a<0)) *' \
                       f'{self.name}_SE2 * {self.name}_E2 * ({self.name}_a - 1)**2 / ' \
                       f'({self.name}_E1 - {self.name}_E2)** 2'
