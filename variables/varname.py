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
        """resize (extend) the list for variable names"""
        yext = self.system.DAE.m - len(self.unamey)
        xext = self.system.DAE.n - len(self.unamex)
        if yext > 0:
            self.unamey.extend([''] * yext)
            self.fnamey.extend([''] * yext)
        if xext > 0:
            self.unamex.extend([''] * xext)
            self.fnamex.extend([''] * xext)

    def append(self, vartype, xy_idx, var_name, element_name):
        """append variable names to the name lists"""
        u = []
        f = []
        if vartype not in ['x', 'y']:
            self.system.Log.Error('Variable type must be ''x'' or ''y'' ')
            return
        elif vartype == 'x':
            u = 'unamex'
            f = 'fnamex'
        elif vartype == 'y':
            u = 'umamey'
            f = 'fnamey'

        self.resize()

        if isinstance(element_name, list) or (BLIST and isinstance(element_name, blist)):
            for i, j in zip(xy_idx, element_name):
                self.__dict__[u][i] = '{0}_{1}'.format(var_name, j)
                self.__dict__[f][i] = '#{0}_{{{1}}}#'.format(var_name, j)
        elif isinstance(element_name, int):
            self.__dict__[u][xy_idx] = '{0}_{1}'.format(var_name, element_name)
            self.__dict__[f][xy_idx] = '#{0}_{{{1}}}#'.format(var_name, element_name)
        else:
            self.system.Log.Warning('Unknown element_name type while building VarName')
