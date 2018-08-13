from cvxopt import matrix
from numpy import array


class VarOut(object):
    """
    Output variable value recorder
    """
    def __init__(self, system):
        """Constructor of empty Varout object"""
        self.system = system
        self.t = []
        self.k = []
        self.vars = []
        self.dat = None
        self._mode = 'w'

    def store(self, t, step):
        """
        Record the state/algeb values at time t to self.vars
        """
        max_cache = int(self.system.TDS.max_cache)
        if len(self.vars) >= max_cache > 0:
            self.dump()
            self.vars = list()
            self.t = list()
            self.k = list()
            self.system.Log.debug('VarOut cache cleared at simulation t = {:g}.'.format(self.system.DAE.t))
            self._mode = 'a'

        self.t.append(t)
        self.k.append(step)
        self.vars.append(matrix([self.system.DAE.x, self.system.DAE.y]))

        if self.system.TDS.compute_flows:
            self.system.DAE.y = self.system.DAE.y[:self.system.DAE.m]

        # self.system.EAGC_module.stream_to_geovis()

    def show(self):
        """
        The representation of an Varout object

        :return: the full result matrix (for use with PyCharm viewer)
        :rtype: array
        """
        out = []

        for item in self.vars:
            out.append(list(item))

        return array(out)

    def dump(self):
        """
        Dump the TDS results to the output `dat` file

        :param lst: dump ``.lst`` file

        :return: succeed flag
        """
        ret = False

        if self.system.Files.no_output:
            # return ``True`` because it did not fail
            return True

        if self.write_lst() and self.write_dat():
            ret = True

        return ret

    def write_dat(self):
        """
        Write ``system.Varout.vars`` to a ``.dat`` file
        :return:
        """
        ret = False

        # compute the total number of columns, excluding time
        n_vars = self.system.DAE.m + self.system.DAE.n
        if self.system.TDS.compute_flows:
            n_vars += 2 * self.system.Bus.n + 4 * self.system.Line.n

        template = ['{:<8g}'] + ['{:0.10f}'] * n_vars
        template = ' '.join(template)

        # format the output in a string
        out = ''
        for t, line in zip(self.t, self.vars):
            values = [t] + list(line)
            out += template.format(*values) + '\n'

        try:
            with open(self.system.Files.dat, self._mode) as f:
                f.write(out)
            ret = True

        except IOError:
            self.system.Log.error('I/O Error while writing the dat file.')

        return ret

    def write_lst(self):
        """
        Dump the variable name lst file

        :return: succeed flag
        """

        ret = False
        out = ''
        varname = self.system.VarName
        template = '{:>6g}, {:>25s}, {:>25s}\n'

        # header line
        out += template.format(0, 'Time [s]', '$Time\ [s]$')

        # output state variables
        for i in range(self.system.DAE.n):
            out += template.format(i + 1, varname.unamex[i], varname.fnamex[i])

        # include line flow variables in algebraic variables
        nflows = 0
        if self.system.TDS.compute_flows:
            nflows = 2 * self.system.Bus.n + 4 * self.system.Line.n + 2 * self.system.Area.n_combination

        # output algebraic variables
        for i in range(self.system.DAE.m + nflows):
            out += template.format(i + 1 + self.system.DAE.n, varname.unamey[i], varname.fnamey[i])

        try:
            with open(self.system.Files.lst, 'w') as f:
                f.write(out)
            ret = True
        except IOError:
            self.system.Log.error('I/O Error while writing the lst file.')

        return ret
