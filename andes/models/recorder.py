from .base import ModelBase


class Recorder(ModelBase):
    """
    Recorder class that outputs simulation data of the given model, variable
    and elements
    """

    def define(self):
        self._group = 'Output'
        self.param_remove('Sn')
        self.param_remove('Vn')
        self.param_remove('fn')

        self.param_define(
            'model',
            default='ALL',
            tomatrix=False,
            descr='model to record',
            mandatory=True)
        self.param_define(
            'variable',
            default='ALL',
            tomatrix=False,
            descr='variable to record',
            mandatory=False)
        self.param_define(
            'element',
            default='ALL',
            tomatrix=False,
            descr='element idx to record',
            mandatory=False)

        self.calls.update({'init1': True})

        self.varout_idx = []
        self.varout_state_idx = []
        self.varout_algeb_idx = []
        self._init()

    def elem_add(self, idx=None, name=None, **kwargs):
        super(Recorder, self).elem_add(idx, name, **kwargs)

    def init1(self, dae):
        if not self.n:
            return

        # include line flow variables in algebraic variables
        nflows = 0
        if self.system.tds.config.compute_flows:
            nflows = 2 * self.system.Bus.n + \
                     4 * self.system.Line.n + \
                     2 * self.system.Area.n_combination

        if 'ALL' in self.model:
            self.varout_idx = list(range(dae.m + dae.n + nflows))
            return

        for model, variable, element in zip(self.model, self.variable,
                                            self.element):
            if isinstance(model, str):
                model = [model]

            # for each model
            for m in model:
                assert m in self.system.devman.devices

                m_instance = self.system.__dict__[m]
                offset = [0]
                if variable == 'ALL':
                    variable = m_instance._states + m_instance._algebs
                    offset = [0] * len(m_instance._states) + [
                        self.system.dae.n
                    ] * len(m_instance._algebs)
                elif isinstance(variable, str):
                    if variable in m_instance._algebs:
                        offset = [self.system.dae.n]
                    variable = [variable]

                # for each variable
                for v, offs in zip(variable, offset):
                    if element == 'ALL':
                        idx = self.system.__dict__[m].get_field(v)
                    else:
                        idx = self.system.__dict__[m].get_field(v, idx=element)

                    idx = [i + offs for i in idx]

                    self.varout_idx.extend(idx)
