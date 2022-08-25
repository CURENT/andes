"""
Module with extra SymPy classes.
"""

import sympy


class Indicator(sympy.Expr):
    """
    Indicator class for printing SymPy Relational.

    Relational expressions in SymPy need to be wrapped by `Indicator`.

    Examples
    --------
    To compare ``dae_t`` with ``0``, one need to use
    ``Indicator(dae_t < 0)```.
    """

    def _numpycode(self, printer):
        return printer._print(self.args[0])

    def _julia(self, printer):
        return "(" + printer._print(self.args[0]) + ")"


class FixPiecewise(sympy.Piecewise):
    """
    A derived Piecewise that fixes the printing of ``select`` to allow compilation with numba.

    See: https://github.com/sympy/sympy/issues/15014
    """

    def _numpycode(self, printer):
        """
        Updated numpy code printer.
        """

        def broadcastarg(arg):
            if arg.has(sympy.Symbol):
                return printer._print(arg)
            if arg == 0:
                return '__zeros'

            return printer._print(arg * sympy.Symbol('__ones'))

        def broadcastcond(cond):
            if cond.has(sympy.Symbol):
                return printer._print(cond)

            if bool(cond):
                return '__trues'
            else:
                return '__falses'

        # Piecewise function printer

        exprs = '[{}]'.format(','.join(broadcastarg(arg.expr) for arg in self.args))
        conds = '[{}]'.format(','.join(broadcastcond(arg.cond) for arg in self.args))
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in self.args.
        # If this is not the case, it may be triggered prematurely.
        return '{}({}, {}, default={})'.format(
            printer._module_format('select'), conds, exprs,
            printer._print(sympy.S.NaN))
