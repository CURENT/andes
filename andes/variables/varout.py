from cvxopt import matrix
from numpy import array


class VarOut(object):
    """Output variable value recorder"""
    def __init__(self, system):
        """Constructor of empty Varout object"""
        self.system = system
        self.t = []
        self.vars = []
        self.dat = None

    def store(self, t):
        """Record the state/algeb values at time t to self.vars"""
        self.t.append(t)
        self.vars.append(matrix([self.system.DAE.x, self.system.DAE.y]))

    def __repr__(self):
        """
        The representation of an Varout object
        :rtype: array
        :return: the full result matrix (for use with PyCharm viewer)
        """
        nvar = self.system.DAE.m + self.system.DAE.n
        nstep = len(self.t)
        return array(self.vars, (nvar, nstep), 'd')

    def dump(self):
        """Dump the TDS results to files after the simulation """
        if self.system.Files.no_output:
            return
        self._write_lst()

        nvars = self.system.DAE.m + self.system.DAE.n + 1
        try:
            self.dat = open(self.system.Files.dat, 'w')
            self.dat.write('{}'.format(nvars) + '\n')

            for t, line in zip(self.t, self.vars):
                self._write_vars(t, line)
            self.dat.close()
        except IOError:
            self.system.Log.error('I/O Error when dumping the dat file.')

    def _write_lst(self):
        """Dump the variable name lst file"""
        try:
            lst = open(self.system.Files.lst, 'w')
            line = '{:>6s}, {:>25s}, {:>25s}\n'.format('0', 'Time [s]', '# Time [s]#')
            lst.write(line)

            varname = self.system.VarName
            for i in range(self.system.DAE.n):
                line = '{:>6g}, {:>25s}, {:>25s}\n'.format(i + 1, varname.unamex[i], varname.fnamex[i])
                lst.write(line)

            for i in range(self.system.DAE.m):
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
