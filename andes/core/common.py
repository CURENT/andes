class ModelFlags(object):
    """
    Model flags.

    Parameters
    ----------
    collate : bool
        True: collate variables by device; False: by variable.
        Non-collate (continuous memory) has faster computation speed.
    pflow : bool
        True: called during power flow
    tds : bool
        True if called during tds; if is False, ``dae_t`` cannot be used
    series : bool
        True if is series device
    nr_iter : bool
        True if is series device
    f_num : bool
        True if the model defines `f_numeric`
    g_num : bool
        True if the model defines `g_numeric`
    j_num : bool
        True if the model defines `j_numeric`
    s_num : bool
        True if the model defines `s_numeric`
    sv_num : bool
        True if the model defines `s_numeric_var`
    """

    def __init__(self, collate=False, pflow=False, tds=False, series=False,
                 nr_iter=False, f_num=False, g_num=False, j_num=False,
                 s_num=False, sv_num=False):

        self.collate = collate
        self.pflow = pflow
        self.tds = tds
        self.series = series
        self.nr_iter = nr_iter
        self.f_num = f_num
        self.g_num = g_num
        self.j_num = j_num
        self.s_num = s_num
        self.sv_num = sv_num
        self.sys_base = False
        self.address = False
        self.initialized = False

    def update(self, dct):
        self.__dict__.update(dct)


class DummyValue(object):
    """
    Class for converting a scalar value to a dummy parameter with `name` and `tex_name` fields.

    A DummyValue object can be passed to Block, which utilizes the `name` field to dynamically generate equations.

    Notes
    -----
    Pass a numerical value to the constructor for most use cases, especially when passing as a v-provider.
    """
    def __init__(self, value):
        self.name = value
        self.tex_name = value
        self.v = value


def dummify(param):
    """
    Dummify scalar parameter and return a DummyValue object. Do nothing for BaseParam instances.

    Parameters
    ----------
    param : float, int, BaseParam
        parameter object or scalar value

    Returns
    -------
    DummyValue(param) if param is a scalar; param itself, otherwise.

    """
    if isinstance(param, (int, float)):
        return DummyValue(param)
    else:
        return param
