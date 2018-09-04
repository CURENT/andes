import logging

logger = logging.getLogger(__name__)


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
        yext = self.system.dae.m - len(self.unamey)
        xext = self.system.dae.n - len(self.unamex)
        if yext > 0:
            self.unamey.extend([''] * yext)
            self.fnamey.extend([''] * yext)
        if xext > 0:
            self.unamex.extend([''] * xext)
            self.fnamex.extend([''] * xext)

    def resize_for_flows(self):
        """Extend `unamey` and `fnamey` for bus injections and line flows"""
        if self.system.tds.config.compute_flows:
            nflows = 2 * self.system.Bus.n + \
                     4 * self.system.Line.n + \
                     2 * self.system.Area.n_combination
            self.unamey.extend([''] * nflows)
            self.fnamey.extend([''] * nflows)

    def append(self, listname, xy_idx, var_name, element_name):
        """Append variable names to the name lists"""
        self.resize()
        string = '{0} {1}'
        if listname not in ['unamex', 'unamey', 'fnamex', 'fnamey']:
            logger.error('Wrong list name for varname.')
            return
        elif listname in ['fnamex', 'fnamey']:
            string = '${0}\ {1}$'

        if isinstance(element_name, list):
            for i, j in zip(xy_idx, element_name):
                # manual elem_add LaTex space for auto-generated element name
                if listname == 'fnamex' or listname == 'fnamey':
                    j = j.replace(' ', '\ ')
                self.__dict__[listname][i] = string.format(var_name, j)
        elif isinstance(element_name, int):
            self.__dict__[listname][xy_idx] = string.format(
                var_name, element_name)
        else:
            logger.warning(
                'Unknown element_name type while building varname')

    def bus_line_names(self):
        """Append bus injection and line flow names to `varname`"""
        if self.system.tds.config.compute_flows:
            self.system.Bus._varname_inj()
            self.system.Line._varname_flow()
            self.system.Area._varname_inter()

    @property
    def uname(self):
        """
        Return the full unformatted variable name list following the order of
        state vars, algeb vars and line flow vars

        :return: list of unformatted names
        """
        return self.unamex + self.unamey

    @property
    def fname(self):
        """
        Return the full formatted variable name list following the order of
        state vars, algeb vars and line flow vars

        :return: list of formatted names
        """
        return self.fnamex + self.fnamey

    def get_xy_name(self, yidx, xidx=0):
        """
        Return variable names for the given indices

        :param yidx:
        :param xidx:
        :return:
        """
        assert isinstance(xidx, int)
        if isinstance(yidx, int):
            yidx = [yidx]

        uname = ['Time [s]'] + self.uname
        fname = ['$Time\ [s]$'] + self.fname

        xname = [list(), list()]
        yname = [list(), list()]

        xname[0] = uname[xidx]
        xname[1] = fname[xidx]

        yname[0] = [uname[i] for i in yidx]
        yname[1] = [fname[i] for i in yidx]

        return xname, yname
