.. _misc:

**********************
Miscellaneous
**********************

.. _per_unit_system:

Per Unit System
===============

The bases for AC system are

- :math:`S_b^{ac}`: three-phase power in MVA. By default, :math:`S_b^{ac}=100 MVA` (set by ``System.config.mva``).

- :math:`V_b^{ac}`: phase-to-phase voltage in kV.

- :math:`I_b^{ac}`: current base :math:`I_b^{ac} = \frac{S_b^{ac}} {\sqrt{3} V_b^{ac}}`

The bases for DC system are

- :math:`S_b^{dc}`: power in MVA. It is assumed to be the same as :math:`S_b^{ac}`.

- :math:`V_b^{dc}`: voltage in kV.

Some device parameters are given as per unit values under the device base power and voltage (if applicable).
For example, the Line model :py:mod:`andes.models.line.Line` has parameters ``r``, ``x`` and ``b``
as per unit values in the device bases ``Sn``, ``Vn1``, and ``Vn2``.
It is up to the user to check data consistency.
For example, line voltage bases are typically the same as bus nominal values.
If the ``r``, ``x`` and ``b`` are meant to be per unit values under the system base,
each Line device should use an ``Sn`` equal to the system base mva.

Parameters in device base will have a property value in the Model References page.
For example, ``Line.r`` has a property ``z``, which means it is a per unit impedance
in the device base.
To find out all applicable properties, refer to the "Other Parameters" section of
:py:mod:`andes.core.param.NumParam`.

After setting up the system, these parameters will be converted to per units
in the bases of system base MVA and bus nominal voltages.
The parameter values in the system base will be stored to the ``v`` attribute of the ``NumParam``.
The original inputs in the device base will be moved to the ``vin`` attribute.
For example, after setting up the system, ``Line.x.v`` is the line reactances in per unit
under system base.

Values in the ``v`` attribute is what get utilized in computation.
Writing new values directly to ``vin`` will not affect the values in ``v`` afterwards.
To alter parameters after setting up, refer to example notebook 2.

Notes
=====

Modeling Blocks
---------------

State Freeze
````````````

State freeze is used by converter controllers during fault transients
to fix a variable at the pre-fault values. The concept of state freeze
is applicable to both state or algebraic variables.
For example, in the renewable energy electric control model (REECA),
the proportional-integral controllers for reactive power error and voltage
error are subject to state freeze when voltage dip is observed.
The internal and output states should be frozen when the freeze signal
turns one and freed when the signal turns back to zero.

Freezing a state variable can be easily implemented by multiplying the freeze
signal with the right-hand side (RHS) of the differential equation:

.. math ::
    T \dot{x} = (1 - z_f) \times f(x)

where :math:`f(x)` is the original RHS of the differential equation,
and :math:`z_f` is the freeze signal. When :math:`z_f` becomes zero
the differential equation will evaluate to zero, making the increment
zero.

Freezing an algebraic variable is more complicate to implement.
One might consider a similar solution to freezing a differential variable
by constructing a piecewise equation, for example,

.. math::
    0 = (1 - z_f)\times g(y)

where :math:`g(y)` is the original RHS of the algebraic equation.
One might also need to add a small value to the diagonals of ``dae.gy``
associated with the algebraic variable to avoid singularity.
The rationale behind this implementation is to zero out the algebraic
equation mismatch and thus stop incremental correction:
in the frozen state, since :math:`z_f` switches to zero,
the algebraic increment should be forced to zero.
This method, however, would not work when a dishonest Newton method is
used.

If the Jacobian matrix is not updated after :math:`z_f` switches to one,
in the row associated with the equation, the derivatives will remain the
same. For the algebraic equation of the PI controller given by

.. math::

    0 = (K_p u + x_i) - y

where :math:`K_p` is the proportional gain, :math:`u` is the input,
:math:`x_I` is the integrator output, and :math:`y` is the PI controller
output, the derivatives w.r.t :math:`u`, :math:`x_i` and :math:`y` are
nonzero in the pre-frozen state. These derivative corrects :math:`y`
following the changes of :math:`u` and :math:`x`.
Although :math:`x` has been frozen, if the Jacobian is not rebuilt,
correction will still be made due to the change of :math:`u`.
Since this equation is linear, only one iteration is needed to let
:math:`y` track the changes of :math:`u`.
For nonlinear algebraic variables, this approach will likely give wrong
results, since the residual is pegged at zero.

To correctly freeze an algebraic variable, the freezing signal needs to
be passed to an ``EventFlag``, which will set an ``custom_event`` flag
if any input changes.
``EventFlag`` is a ``VarService`` that will be evaluated at each
iteration after discrete components and before equations.


Profiling Import
========================================
To speed up the command-line program, import profiling is used to breakdown the program loading time.

With tool ``profimp``, ``andes`` can be profiled with ``profimp "import andes" --html > andes_import.htm``. The
report can be viewed in any web browser.

What won't not work
===================

You might have heard that PyPy is faster than CPython due to a built-in JIT compiler.
Before you spend an hour compiling the dependencies, here is the fact:
PyPy won't work for speeding up ANDES.

PyPy is often much slower than CPython when using CPython extension modules
(see the PyPy_FAQ_).
Unfortunately, NumPy is one of the highly optimized libraries that heavily
use CPython extension modules.

.. _PyPy_FAQ: https://doc.pypy.org/en/latest/faq.html#do-c-extension-modules-work-with-pypy
