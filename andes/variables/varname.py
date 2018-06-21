try:
    from blist import *
    BLIST = True
except ImportError:
    BLIST = False


class VarName(object):
    """Variable name manager class"""
    def __init__(self, system):
        self.system = system
        self.unamex = []  # unformatted state variable names
        self.unamey = []  # unformatted algeb variable names
        self.fnamex = []  # formatted state variable names
        self.fnamey = []  # formatted algeb variable names

    def resize(self):
        """Resize (extend) the list for variable names"""
        yext = self.system.DAE.m - len(self.unamey)
        xext = self.system.DAE.n - len(self.unamex)
        if yext > 0:
            self.unamey.extend([''] * yext)
            self.fnamey.extend([''] * yext)
        if xext > 0:
            self.unamex.extend([''] * xext)
            self.fnamex.extend([''] * xext)

    def resize_for_flows(self):
        """Extend `unamey` and `fnamey` for bus injections and line flows"""

        if self.system.TDS.compute_flows:
            nflows = 2 * self.system.Bus.n + 4 * self.system.Line.n + 2 * self.system.Area.n_combination  # added areas
            self.unamey.extend([''] * nflows)
            self.fnamey.extend([''] * nflows)

    def append(self, listname, xy_idx, var_name, element_name):
        """Append variable names to the name lists"""
        self.resize()
        string = '{0} {1}'
        if listname not in ['unamex', 'unamey', 'fnamex', 'fnamey']:
            self.system.Log.error('Wrong list name for VarName.')
            return
        elif listname in ['fnamex', 'fnamey']:
            string = '${0}\ {1}$'

        if isinstance(element_name, list) or (BLIST and isinstance(element_name, blist)):
            for i, j in zip(xy_idx, element_name):
                # manual add LaTex space for auto-generated element name
                if listname == 'fnamex' or listname == 'fnamey':
                    j = j.replace(' ', '\ ')
                self.__dict__[listname][i] = string.format(var_name, j)
        elif isinstance(element_name, int):
            self.__dict__[listname][xy_idx] = string.format(var_name, element_name)
        else:
            self.system.Log.warning('Unknown element_name type while building VarName')

    def bus_line_names(self):
        """Append bus injection and line flow names to `VarName`"""
        if self.system.TDS.compute_flows:
            self.system.Bus._varname_inj()
            self.system.Line._varname_flow()
            self.system.Area._varname_inter()
