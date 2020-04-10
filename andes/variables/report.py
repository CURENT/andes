import logging
from time import strftime
from collections import OrderedDict

from andes import __version__ as version
from andes.io.txt import dump_data
from andes.utils.misc import elapsed
from andes.shared import np

logger = logging.getLogger(__name__)
all_formats = {}


class Report(object):
    """
    Report class to store system static analysis reports
    """

    def __init__(self, system):
        self.system = system
        self.basic = OrderedDict()
        self.extended = OrderedDict()

    @property
    def info(self):
        system = self.system
        info = list()
        info.append('ANDES' + ' ' + version + '\n')
        info.append('Copyright (C) 2015-2020 Hantao Cui\n\n')
        info.append('ANDES comes with ABSOLUTELY NO WARRANTY\n')
        info.append('Case file: ' + system.files.case + '\n')
        info.append('Report time: ' + strftime("%m/%d/%Y %I:%M:%S %p") + '\n\n')
        if system.PFlow.converged is True:
            info.append(f'Power flow converged in {system.PFlow.niter} iterations.\n')
            info.append('Flat-start: ' +
                        ('Yes' if system.Bus.config.flat_start else 'No') + '\n')

        return info

    def update(self):
        """
        Update values based on the requested content
        """
        system = self.system
        self.basic.update({
            'Buses': system.Bus.n,
            'Generators': system.PV.n + system.Slack.n,
            'Committed Gens': int(sum(system.PV.u.v) + sum(system.Slack.u.v)),
            'Loads': system.PQ.n,
            'Shunts': system.Shunt.n,
            'Lines': system.Line.n,
            'Transformers': np.count_nonzero(system.Line.trans.v == 1),
            'Areas': system.Area.n,
        })

        if self.system.PFlow.converged is False:
            logger.warning('Cannot update extended summary. Power flow not solved.')
            return

        self.extended.update({
            'Pg': sum(system.PV.u.v * system.PV.p.v) + sum(system.Slack.u.v * system.Slack.p.v),
            'Qg': sum(system.PV.u.v * system.PV.q.v) + sum(system.Slack.u.v * system.Slack.q.v),
            'Pl': round(float(sum(system.PQ.p0.v)), 6),
            'Ql': round(float(sum(system.PQ.q0.v)), 6),
            'Ptot': sum(system.PV.pmax.v) + sum(system.Slack.pmax.v),
            'Pon': sum(system.PV.u.v * system.PV.pmax.v),
            'Qtot_min': sum(system.PV.qmin.v) + sum(system.Slack.qmin.v),
            'Qtot_max': sum(system.PV.qmax.v) + sum(system.Slack.qmax.v),
            'Qon_min': sum(system.PV.u.v * system.PV.qmin.v),
            'Qon_max': sum(system.PV.u.v * system.PV.qmax.v),
        })

    def write(self):
        """
        Write report to file.
        """
        system = self.system
        if system.files.no_output is True:
            return

        text = list()
        header = list()
        row_name = list()
        data = list()
        self.update()

        t, _ = elapsed()

        # ----------------------------------------
        # info section
        text.append(self.info)
        header.append(None)
        row_name.append(None)
        data.append(None)
        # ----------------------------------------

        # ----------------------------------------
        # summary section
        text.append(['Statistics:\n'])
        header.append(None)
        row_name.append(self.basic.keys())
        data.append(list(self.basic.values()))
        # ----------------------------------------

        if len(self.extended):
            text.append(['EXTENDED SUMMARY:\n'])
            header.append(['P (pu)', 'Q (pu)'])
            row_name.append(
                ['Generation', 'Load'])
            Pcol = [
                self.extended['Pg'],
                self.extended['Pl'],
            ]

            Qcol = [
                self.extended['Qg'],
                self.extended['Ql'],
            ]

            data.append([Pcol, Qcol])

        if system.PFlow.converged:

            # ----------------------------------------
            # Bus data
            text.append(['BUS DATA:\n'])
            header.append(['Vm(pu)', 'Va(rad)'])
            row_name.append(system.Bus.name.v)
            data.append([system.Bus.v.v, system.Bus.a.v])
            # ----------------------------------------

            # ----------------------------------------
            # Node data
            if hasattr(system, 'Node') and system.Node.n:
                text.append(['NODE DATA:\n'])
                header.append(['V(pu)'])
                row_name.append(system.Node.name.v)
                data.append([system.Node.v.v])
            # ----------------------------------------

            # ----------------------------------------
            # Line data
            text.append(['LINE DATA:\n'])
            header.append([
                'From Bus (idx)', 'To Bus (idx)', 'P From (pu)', 'Q From (pu)',
                'P To (pu)', 'Q To(pu)'
            ])
            row_name.append(system.Line.name.v)
            data.append([system.Line.bus1.v,
                         system.Line.bus2.v,
                         system.Line.a1.e,
                         system.Line.v1.e,
                         system.Line.a2.e,
                         system.Line.v2.e
                         ])
            # ----------------------------------------

            # ----------------------------------------
            # Additional Algebraic data
            text.append(['OTHER ALGEBRAIC VARIABLES:\n'])
            header.append([''])
            row_name.append(system.dae.y_name[2 * system.Bus.n:system.dae.m])
            data.append([round(i, 6) for i in system.dae.y[2 * system.Bus.n:]])

            # Additional State variable data
            if system.dae.n:
                text.append(['OTHER STATE VARIABLES:\n'])
                header.append([''])
                row_name.append(system.dae.x_name[:])
                data.append([round(i, 6) for i in system.dae.x[:]])

        dump_data(text, header, row_name, data, system.files.output)

        _, s = elapsed(t)
        logger.info(f'Report saved to "{system.files.output}" in {s}.')
