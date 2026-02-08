#  [ANDES] (C)2015-2025 Hantao Cui
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.

"""
Observable variable class for explicit algebraic assignments.
"""

from typing import Optional

import numpy as np

from andes.core.var import BaseVar


class Observable(BaseVar):
    """
    A variable computed by explicit assignment rather than solved
    in the DAE system.

    At code generation time, the Observable's expression is substituted
    into all referencing equations so that Jacobians are correctly
    computed via direct differentiation. Post-solve, values are evaluated
    and stored in ``dae.b`` for recording.

    Parameters
    ----------
    e_str : str, optional
        Assignment expression string. Unlike ``Algeb`` where ``e_str``
        means residual (``0 = e_str``), for Observable it means direct
        assignment (``b = e_str``).

    Examples
    --------
    To define an observable ``vd`` computed from bus voltage and angle:

    .. code-block:: python

        self.vd = Observable(e_str='v * cos(delta - a)',
                             info='d-axis voltage',
                             tex_name='V_d')

    The symbol ``vd`` can then be used in other equations. At code
    generation time, ``vd`` will be replaced by ``v * cos(delta - a)``
    in those equations.

    Attributes
    ----------
    e_code : None
        Observable has no equation residual.
    v_code : str
        Variable code string, equals string literal ``b``.
    """

    e_code = None
    v_code = 'b'

    def __init__(self,
                 e_str: Optional[str] = None,
                 name: Optional[str] = None,
                 tex_name: Optional[str] = None,
                 info: Optional[str] = None,
                 unit: Optional[str] = None,
                 discrete=None,
                 ):
        super().__init__(name=name, tex_name=tex_name, info=info, unit=unit,
                         e_str=e_str, discrete=discrete)

    def set_address(self, addr: np.ndarray, contiguous=False):
        """
        Set the address of this Observable in ``dae.b``.

        Overrides BaseVar to ensure ``e_inplace`` is always False
        (no equation array for Observable).
        """
        self.a = addr
        self.n = len(self.a)
        self._contiguous = contiguous

        # Observable values can be in-place views into dae.b
        if self._contiguous:
            self.v_inplace = True

        # Never in-place for equation (no equation array)
        self.e_inplace = False

    def set_arrays(self, dae, inplace=True, alloc=True):
        """
        Set the value array for this Observable.

        Only sets ``self.v`` (into ``dae.b``). No equation array is
        allocated since Observable has no residual.
        """
        if inplace and self.v_inplace and self.n > 0:
            slice_idx = slice(self.a[0], self.a[-1] + 1)
            self.v = dae.__dict__[self.v_code][slice_idx]
        elif alloc:
            if not self.v_inplace:
                self.v = np.zeros(self.n)

        # No equation array — skip e allocation entirely

    def reset(self):
        """
        Reset the internal numpy arrays and flags.
        """
        self.n = 0
        self.a[:] = 0
        self.v[:] = 0
        self._contiguous = False
        self.v_inplace = False
        self.e_inplace = False
