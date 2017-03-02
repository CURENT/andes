import platform
from operator import itemgetter
import importlib
from cvxopt.base import mul
from ..formats import all_formats
from time import strftime

revision = '2017.03.01'
this_year = revision[:4]


def preamble(disable=False):
    if disable:
        return
    message = '\n'
    message += 'ANDES ' + revision + '\n'
    message += 'Copyright (C) 2015-' + this_year + ' Hantao Cui\n\n'
    message += 'ANDES comes with ABSOLUTELY NO WARRANTY\n'
    message += 'Use this software AT YOUR OWN RISK\n\n'
    message += 'Platform:    ' + platform.system() + '\n'
    message += 'Interpreter: ' + 'Python ' + platform.python_version() + '\n'
    message += 'Session:     ' + strftime("%m/%d/%Y %I:%M:%S %p") + '\n\n'

    return message


class Report(object):
    """Report class to store system static analysis reports"""
    def __init__(self, system):
        self.system = system
        self.basic = {}
        self.extended = {}
        self.powerflow = []

        self._basic = ['nbus', 'ngen', 'ngen_on', 'nload', 'nshunt', 'nline', 'ntransf', 'narea']
        self._basic_name = ['Buses', 'Generators', 'Committed Gens', 'Loads', 'Shunts', 'Lines', 'Transformers', 'Areas']

        self._extended = ['Ptot', 'Pon', 'Pg', 'Qtot_min', 'Qtot_max', 'Qon_min', 'Qon_max', 'Qg', 'Pl', 'Ql',
                          'Psh', 'Qsh', 'Ploss', 'Qloss', 'Pch', 'Qch']
        self.build_dict()

    def build_dict(self):
        for item in self._basic:
            self.basic[item] = 0.0
        for item in self._extended:
            self.extended[item] = 0.0

    def update_pf(self):
        """Update power flow solution to system.Report"""
        self.update_summary()
        if self.system.Settings.pfsolved is True:
            self.update_extended()

    def _update(self, item, key, val):
        """helper function to update a key in the dict"""
        self.__dict__[item][key] = val

    def update_summary(self):
        system = self.system
        self.basic.update({'nbus':    system.Bus.n,
                           'ngen':    system.PV.n + system.SW.n,
                           'ngen_on': sum(system.PV.u) + sum(system.SW.u),
                           'nload':   system.PQ.n,
                           'nshunt':  system.Shunt.n,
                           'nline':   system.Line.n,
                           'ntransf': system.Line.trasf.count(True),
                           'narea':   system.Area.n,
                           })

    def update_extended(self):
        system = self.system
        Sloss = sum(system.Line.Sfr + system.Line.Sto)
        self.extended.update({'Ptot': sum(system.PV.pmax) + sum(system.SW.pmax),  # + sum(system.SW.pmax)
                              'Pon': sum( mul(system.PV.u, system.PV.pmax) ),
                              'Pg': sum(system.Bus.Pg),
                              'Qtot_min': sum(system.PV.qmin) + sum(system.SW.qmin),
                              'Qtot_max': sum(system.PV.qmax) + sum(system.SW.qmax),
                              'Qon_min':  sum(mul(system.PV.u, system.PV.qmin)),
                              'Qon_max':  sum(mul(system.PV.u, system.PV.qmax)),
                              'Qg': round(sum(system.Bus.Qg), 5),
                              'Pl': round(sum(system.PQ.p), 5),
                              'Ql': round(sum(system.PQ.q), 5),
                              'Psh': 0.0,
                              'Qsh': round(sum(system.PQ.q) - sum(system.Bus.Ql), 5) ,
                              'Ploss': round(Sloss.real, 5),
                              'Qloss': round(Sloss.imag, 5),
                              'Pch': 0.0,
                              'Qch': round(sum(system.Line.chgfr.real() + system.Line.chgto.real()), 5) ,
                              })

    def write_summary(self):
        system = self.system

        file = system.Files.output
        export = all_formats.get(system.Settings.export, 'txt')
        module = importlib.import_module('andes.formats.' + export)
        dump_data = getattr(module, 'dump_data')

        text = list()
        header = list()
        rowname = list()
        data = list()

        info = list()
        info.append('ANDES' + ' ' + revision + '\n')
        info.append('Copyright (C) 2015-' + this_year + ' Hantao Cui' + '\n\n')
        info.append('Case file: ' + system.Files.case + '\n')
        info.append('Report Time: ' + strftime("%m/%d/%Y %I:%M:%S %p") + '\n\n')
        info.append('Power flow method: ' + system.PF.solver.upper() + '\n')
        info.append('Flat-start: ' + ('Yes' if system.PF.flatstart else 'No') + '\n')

        text.append(info)
        header.append(None)
        rowname.append(None)
        data.append(None)

        # Basic summary
        text.append(['SUMMARY:\n'])
        header.append(None)
        rowname.append(self._basic_name)
        data.append([self.basic[item] for item in self._basic])

        dump_data(text, header, rowname, data, file)
        return

    def write_pf(self):
        system = self.system

        file = system.Files.output
        export = all_formats.get(system.Settings.export, 'txt')
        module = importlib.import_module('andes.formats.' + export)
        dump_data = getattr(module, 'dump_data')

        text = list()
        header = list()
        rowname = list()
        data = list()

        info = list()
        info.append('ANDES' + ' ' + revision + '\n')
        info.append('Copyright (C) 2015-2017 Hantao Cui' + '\n\n')
        info.append('Case file: ' + system.Files.case + '\n')
        info.append('Session: ' + strftime("%m/%d/%Y %I:%M:%S %p") + '\n\n')
        info.append('Power flow method: ' + system.PF.solver.upper() + '\n')
        info.append('Flat-start: ' + ('True' if system.PF.flatstart else 'False') + '\n')

        text.append(info)
        header.append(None)
        rowname.append(None)
        data.append(None)

        # Basic summary
        text.append(['SUMMARY:\n'])
        header.append(None)
        rowname.append(self._basic_name)
        data.append([self.basic[item] for item in self._basic])

        # Extended summary
        text.append(['EXTENDED SUMMARY:\n'])
        header.append(['P (pu)', 'Q (pu)'])
        rowname.append(['Generation', 'Load', 'Shunt Inj', 'Losses', 'Line Charging'])
        Pcol = [self.extended['Pg'],
                self.extended['Pl'],
                self.extended['Psh'],
                self.extended['Ploss'],
                self.extended['Pch'],
                ]

        Qcol = [self.extended['Qg'],
                self.extended['Ql'],
                self.extended['Qsh'],
                self.extended['Qloss'],
                self.extended['Qch'],
                ]

        data.append([Pcol, Qcol])

        # Bus data
        idx, name, Vm, Va, Pg, Qg, Pl, Ql = system.get_busdata()
        text.append(['BUS DATA:\n'])
        Va_unit = 'deg' if system.PF.usedegree else 'rad'
        header.append(['Vm(pu)', 'Va({:s})'.format(Va_unit), 'Pg (pu)', 'Qg (pu)', 'Pl (pu)', 'Ql (pu)'])
        name = ['<' + str(i) + '>' + j for i, j in zip(idx, name)]
        rowname.append(name)
        data.append([Vm, Va, Pg, Qg, Pl, Ql])

        # Node data
        if system.Node.n:
            idx, name, V = system.get_nodedata()
            text.append(['NODE DATA:\n'])
            header.append(['V(pu)'])
            rowname.append(name)
            data.append([V])

        # Line data
        name, fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss = system.get_linedata()
        text.append(['LINE DATA:\n'])
        header.append(['From Bus', 'To Bus', 'P From (pu)',  'Q From (pu)', 'P To (pu)', 'Q To(pu)', 'P Loss(pu)', 'Q Loss(pu)'])
        rowname.append(name)
        data.append([fr, to, Pfr, Qfr, Pto, Qto, Ploss, Qloss])

        # Additional Algebraic data
        text.append(['OTHER ALGEBRAIC VARIABLES:\n'])
        header.append([''])
        rowname.append(system.Varname.unamey[2*system.Bus.n:])
        data.append([round(i, 5) for i in system.DAE.y[2*system.Bus.n:]])

        # Additional State variable data
        if system.DAE.n:
            text.append(['OTHER STATE VARIABLES:\n'])
            header.append([''])
            rowname.append(system.Varname.unamex[:])
            data.append([round(i, 5) for i in system.DAE.x[:]])

        dump_data(text, header, rowname, data, file)
        return
