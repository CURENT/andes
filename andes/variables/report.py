import importlib
from time import strftime

from cvxopt import mul

from ..formats import all_formats
from ..utils import elapsed
from andes import __version__ as version

import logging
logger = logging.getLogger(__name__)


class Report(object):
    """
    Report class to store system static analysis reports
    """

    def __init__(self, system):
        self.system = system
        self.basic = {}
        self.extended = {}

        self._basic = [
            'nbus', 'ngen', 'ngen_on', 'nload', 'nshunt', 'nline', 'ntransf',
            'narea'
        ]
        self._basic_name = [
            'Buses', 'Generators', 'Committed Gens', 'Loads', 'Shunts',
            'Lines', 'Transformers', 'Areas'
        ]

        self._extended = [
            'Ptot', 'Pon', 'Pg', 'Qtot_min', 'Qtot_max', 'Qon_min', 'Qon_max',
            'Qg', 'Pl', 'Ql', 'Psh', 'Qsh', 'Ploss', 'Qloss', 'Pch', 'Qch'
        ]

        for item in self._basic:
            self.basic[item] = 0.0
        for item in self._extended:
            self.extended[item] = 0.0

    @property
    def info(self):
        info = list()
        info.append('ANDES' + ' ' + version + '\n')
        info.append('Copyright (C) 2015-2018 Hantao Cui\n\n')
        info.append('ANDES comes with ABSOLUTELY NO WARRANTY\n')
        info.append('Use this software AT YOUR OWN RISK\n\n')
        info.append('Case file: ' + self.system.files.case + '\n')
        info.append('report Time: ' + strftime("%m/%d/%Y %I:%M:%S %p") +
                    '\n\n')
        if self.system.pflow.solved is True:
            info.append('Power flow method: ' +
                        self.system.pflow.config.method.upper() + '\n')
            info.append('Number of iterations: ' + str(self.system.pflow.config.iter) +
                        '\n')
            info.append('Flat-start: ' +
                        ('Yes' if self.system.pflow.config.flatstart else 'No') + '\n')

        return info

    def _update_summary(self, system):
        """
        Update the summary data

        Parameters
        ----------
        system

        Returns
        -------
        None
        """
        self.basic.update({
            'nbus': system.Bus.n,
            'ngen': system.PV.n + system.SW.n,
            'ngen_on': sum(system.PV.u) + sum(system.SW.u),
            'nload': system.PQ.n,
            'nshunt': system.Shunt.n,
            'nline': system.Line.n,
            'ntransf': system.Line.trasf.count(True),
            'narea': system.Area.n,
        })

    def _update_extended(self, system):
        """Update the extended data"""
        if self.system.pflow.solved is False:
            logger.warning(
                'Cannot update extended summary. Power flow not solved.')
            return

        Sloss = sum(system.Line.S1 + system.Line.S2)
        self.extended.update({
            'Ptot':
            sum(system.PV.pmax) + sum(system.SW.pmax),  # + sum(system.SW.pmax)
            'Pon':
            sum(mul(system.PV.u, system.PV.pmax)),
            'Pg':
            sum(system.Bus.Pg),
            'Qtot_min':
            sum(system.PV.qmin) + sum(system.SW.qmin),
            'Qtot_max':
            sum(system.PV.qmax) + sum(system.SW.qmax),
            'Qon_min':
            sum(mul(system.PV.u, system.PV.qmin)),
            'Qon_max':
            sum(mul(system.PV.u, system.PV.qmax)),
            'Qg':
            round(sum(system.Bus.Qg), 5),
            'Pl':
            round(sum(system.PQ.p), 5),
            'Ql':
            round(sum(system.PQ.q), 5),
            'Psh':
            0.0,
            'Qsh':
            round(sum(system.PQ.q) - sum(system.Bus.Ql), 5),
            'Ploss':
            round(Sloss.real, 5),
            'Qloss':
            round(Sloss.imag, 5),
            'Pch':
            round(sum(system.Line.Pchg1 + system.Line.Pchg2), 5),
            'Qch':
            round(sum(system.Line.Qchg1 + system.Line.Qchg2), 5),
        })

    def update(self, content=None):
        """
        Update values based on the requested content

        Parameters
        ----------
        content

        Returns
        -------

        """
        if not content:
            return
        if content == 'summary' or 'extended' or 'powerflow':
            self._update_summary(self.system)
        if content == 'extended' or 'powerflow':
            self._update_extended(self.system)

    def write(self, content=None):
        """
        Write report to file.

        Parameters
        ----------
        content: str
            'summary', 'extended', 'powerflow'

        """
        if self.system.files.no_output is True:
            return

        t, _ = elapsed()

        if not content:
            logger.warning('report content not specified.')
            return

        self.update(content)

        system = self.system
        file = system.files.output
        export = all_formats.get(system.config.export, 'txt')
        module = importlib.import_module('andes.formats.' + export)
        dump_data = getattr(module, 'dump_data')

        text = list()
        header = list()
        rowname = list()
        data = list()

        text.append(self.info)
        header.append(None)
        rowname.append(None)
        data.append(None)

        if content == 'summary' or 'extended' or 'powerflow':
            text.append(['SUMMARY:\n'])
            header.append(None)
            rowname.append(self._basic_name)
            data.append([self.basic[item] for item in self._basic])

        if content == 'extended' or 'powerflow':
            text.append(['EXTENDED SUMMARY:\n'])
            header.append(['P (pu)', 'Q (pu)'])
            rowname.append(
                ['Generation', 'Load', 'Shunt Inj', 'Losses', 'Line Charging'])
            Pcol = [
                self.extended['Pg'],
                self.extended['Pl'],
                self.extended['Psh'],
                self.extended['Ploss'],
                self.extended['Pch'],
            ]

            Qcol = [
                self.extended['Qg'],
                self.extended['Ql'],
                self.extended['Qsh'],
                self.extended['Qloss'],
                self.extended['Qch'],
            ]

            data.append([Pcol, Qcol])

        if content == 'powerflow' and system.pflow.solved:
            idx, name, Vm, Va, Pg, Qg, Pl, Ql = system.get_busdata()
            Va_unit = 'deg' if system.pflow.config.usedegree else 'rad'
            text.append(['BUS DATA:\n'])
            # todo: consider system.pflow.config.units
            header.append([
                'Vm(pu)', 'Va({:s})'.format(Va_unit), 'Pg (pu)', 'Qg (pu)',
                'Pl (pu)', 'Ql (pu)'
            ])
            name = [str(i) + '-' + j[:8] for i, j in zip(idx, name)]
            rowname.append(name)
            data.append([Vm, Va, Pg, Qg, Pl, Ql])

            # Node data
            if hasattr(system, 'Node') and system.Node.n:
                idx, name, V = system.get_nodedata()
                text.append(['NODE DATA:\n'])
                header.append(['V(pu)'])
                rowname.append(name)
                data.append([V])

            # Line data
            name, fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss = \
                system.get_linedata()
            text.append(['LINE DATA:\n'])
            header.append([
                'From Bus', 'To Bus', 'P From (pu)', 'Q From (pu)',
                'P To (pu)', 'Q To(pu)', 'P Loss(pu)', 'Q Loss(pu)'
            ])
            rowname.append(name)
            data.append([fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss])

            # Additional Algebraic data
            text.append(['OTHER ALGEBRAIC VARIABLES:\n'])
            header.append([''])
            rowname.append(
                system.varname.unamey[2 * system.Bus.n:system.dae.m])
            data.append([round(i, 5) for i in system.dae.y[2 * system.Bus.n:]])

            # Additional State variable data
            if system.dae.n:
                text.append(['OTHER STATE VARIABLES:\n'])
                header.append([''])
                rowname.append(system.varname.unamex[:])
                data.append([round(i, 5) for i in system.dae.x[:]])

        dump_data(text, header, rowname, data, file)

        _, s = elapsed(t)
        logger.info('report written to <{:s}> in {:s}.'.format(system.files.output, s))
