.. _misc:

**********************
Miscellaneous
**********************

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

Per Unit System
==============================

The bases for AC system are

- :math:`S_b^{ac}`: three-phase power in MVA. By default, :math:`S_b^{ac}=100 MVA` (in ``System.config.mva``).

- :math:`V_b^{ac}`: phase-to-phase voltage in kV.

- :math:`I_b^{ac}`: current base :math:`I_b^{ac} = \frac{S_b^{ac}} {\sqrt{3} V_b^{ac}}`

The bases for DC system are

- :math:`S_b^{dc}`: power in MVA. It is assumed to be the same as :math:`S_b^{ac}`.

- :math:`V_b^{dc}`: voltage in kV.

Some device parameters with specific properties are per unit values under the corresponding
device base ``Sn`` and ``Vn`` (if applicable).
These properties are documented in :py:mod:`andes.core.param.NumParam`.

After setting up the system, these parameters will be converted to the system base MVA
as specified in the config file (100 MVA by default).
The parameter values in the system base will be stored in the ``v`` attribute of the ``NumParam``,
and the original inputs in the device base will be stored to the ``vin`` attribute.
Values in the ``v`` attribute is what get utilized in computation.
Writing new values directly to ``vin`` will not affect the values in ``v`` afterwards.

Profiling Import
========================================
To speed up the command-line program, import profiling is used to breakdown the program loading time.

With tool ``profimp``, ``andes`` can be profiled with ``profimp "import andes" --html > andes_import.htm``. The
report can be viewed in any web browser.
