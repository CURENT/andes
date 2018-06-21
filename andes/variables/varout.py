from cvxopt import matrix
from numpy import array


class VarOut(object):
    """Output variable value recorder"""
    def __init__(self, system):
        """Constructor of empty Varout object"""
        self.system = system
        self.t = []
        self.k = []
        self.vars = []
        self.dat = None
        self._mode = 'w'

    def store(self, t, step):
        """Record the state/algeb values at time t to self.vars"""

        if len(self.vars) >= 500:
            self.dump(lst=False)
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
        :rtype: array
        :return: the full result matrix (for use with PyCharm viewer)
        """
        nvar = self.system.DAE.m + self.system.DAE.n
        nstep = len(self.t)
        out = []
        for item in self.vars:
            out.append(list(item))
        return array(out)

    def dump(self, lst=True):
        """Dump the TDS results to the output `dat` file"""
        if self.system.Files.no_output:
            return
        if lst:
            self._write_lst()

        nvars = self.system.DAE.m + self.system.DAE.n + 1
        if self.system.TDS.compute_flows:
            nvars += 2 * self.system.Bus.n + 4 * self.system.Line.n

        try:
            self.dat = open(self.system.Files.dat, self._mode)

            for t, line in zip(self.t, self.vars):
                self._write_vars(t, line)
            self.dat.close()
        except IOError:
            self.system.Log.error('I/O Error when dumping the dat file.')

    def _write_lst(self):
        """Dump the variable name lst file"""
        try:
            lst = open(self.system.Files.lst, 'w')
            line = '{:>6s}, {:>25s}, {:>25s}\n'.format('0', 'Time [s]', '$Time\ [s]$')
            lst.write(line)

            varname = self.system.VarName
            for i in range(self.system.DAE.n):
                line = '{:>6g}, {:>25s}, {:>25s}\n'.format(i + 1, varname.unamex[i], varname.fnamex[i])
                lst.write(line)

            nflows = 0
            if self.system.TDS.compute_flows:
                nflows = 2 * self.system.Bus.n + 4 * self.system.Line.n + 2 * self.system.Area.n_combination

            for i in range(self.system.DAE.m + nflows):
                line = '{:>6g}, {:>25s}, {:>25s}\n'.format(i + 1 + self.system.DAE.n, varname.unamey[i], varname.fnamey[i])
                lst.write(line)

            lst.close()
        except IOError:
            self.system.Log.error('I/O Error when writing the lst file.')

    def _write_vars(self, t, vars):
        """Helper function to write one line of simulation results"""
        nvars = vars.size[0]

        line = ['{:<8g}'] + ['{:0.10f}'] * nvars
        line = ' '.join(line)
        values = [t] + list(vars)
        self.dat.write(line.format(*values) + '\n')
