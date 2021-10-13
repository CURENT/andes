import sympy as sym


class fixPiecewise(sym.Piecewise):
    """
    A derived Piecewise that fixes the printing of ``select`` to allow compilation with numba.

    See: https://github.com/sympy/sympy/issues/15014
    """

    def _numpycode(expr, printer):
        """
        Updated numpy code printer.
        """

        # find any symbol:
        s = list(expr.atoms(sym.Symbol))[0]

        def broadcastarg(arg):
            if arg.has(sym.Symbol):
                return printer._print(arg)
            elif arg == 0:
                return printer._print(sym.Symbol(printer._module+'.zeros_like({0})'.format(s)))
            else:
                return printer._print(arg*sym.Symbol(printer._module+'.ones_like({0})'.format(s)))

        def broadcastcond(cond):
            if cond.has(sym.Symbol):
                return printer._print(cond)
            else:
                return printer._module+'.full({0}.shape,{1})'.format(printer._print(s), printer._print(cond))

        "Piecewise function printer"
        exprs = '[{}]'.format(','.join(broadcastarg(arg.expr) for arg in expr.args))
        conds = '[{}]'.format(','.join(broadcastcond(arg.cond) for arg in expr.args))
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.
        return '{}({}, {}, default={})'.format(
            printer._module_format(printer._module + '.select'), conds, exprs,
            printer._print(sym.S.NaN))
