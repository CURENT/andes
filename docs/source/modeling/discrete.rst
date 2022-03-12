
Discrete
========

Background
----------
The discrete component library contains a special type of block for modeling the discontinuity in power system
devices. Such continuities can be device-level physical constraints or algorithmic limits imposed on controllers.

The base class for discrete components is :py:mod:`andes.core.discrete.Discrete`.

.. currentmodule:: andes.core.discrete
.. autosummary::
      :recursive:
      :toctree: _generated

      Discrete


The uniqueness of discrete components is the way it works.
Discrete components take inputs, criteria, and exports a set of flags with the component-defined meanings.
These exported flags can be used in algebraic or differential equations to build piece-wise equations.

For example, `Limiter` takes a v-provider as input, two v-providers as the upper and the lower bound.
It exports three flags: `zi` (within bound), `zl` (below lower bound), and `zu` (above upper bound).
See the code example in ``models/pv.py`` for an example voltage-based PQ-to-Z conversion.

It is important to note when the flags are updated.
Discrete subclasses can use three methods to check and update the value and equations.
Among these methods, `check_var` is called *before* equation evaluation, but `check_eq` and `set_eq` are
called *after* equation update.
In the current implementation, `check_var` updates flags for variable-based discrete components (such as
`Limiter`).
`check_eq` updates flags for equation-involved discrete components (such as `AntiWindup`).
`set_var`` is currently only used by `AntiWindup` to store the pegged states.

ANDES includes the following types of discrete components.

Limiters
--------
.. autoclass:: andes.core.discrete.Limiter
    :noindex:

.. autoclass:: andes.core.discrete.SortedLimiter
    :noindex:

.. autoclass:: andes.core.discrete.HardLimiter
    :noindex:

.. autoclass:: andes.core.discrete.RateLimiter
    :noindex:

.. autoclass:: andes.core.discrete.AntiWindup
    :noindex:

.. autoclass:: andes.core.discrete.AntiWindupRate
    :noindex:

Comparers
---------
.. autoclass:: andes.core.discrete.LessThan
    :noindex:

.. autoclass:: andes.core.discrete.Selector
    :noindex:

.. autoclass:: andes.core.discrete.Switcher
    :noindex:

Deadband
--------
.. autoclass:: andes.core.discrete.DeadBand
    :noindex:

.. autoclass:: andes.core.discrete.DeadBandRT
    :noindex:


Others
------
.. autoclass:: andes.core.discrete.Average
    :noindex:

.. autoclass:: andes.core.discrete.Delay
    :noindex:

.. autoclass:: andes.core.discrete.Derivative
    :noindex:

.. autoclass:: andes.core.discrete.Sampling
    :noindex:

.. autoclass:: andes.core.discrete.ShuntAdjust
    :noindex:
