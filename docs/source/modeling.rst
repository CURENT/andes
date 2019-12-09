.. _modeling:

**********************
Modeling
**********************

Per Unit System
==============================

The bases for AC system are

- :math:`S_b^{ac}`: three-phase power in MVA. By default, :math:`S_b^{ac}=100 MVA` (in ``System.config.mva``).

- :math:`V_b^{ac}`: phase-to-phase voltage in kV.

- :math:`I_b^{ac}`: current base :math:`I_b^{ac} = \frac{S_b^{ac}} {\sqrt{3} V_b^{ac}}`

The bases for DC system are

- :math:`S_b^{dc}`: power in MVA. It is assumed to be the same as :math:`S_b^{ac}`.

- :math:`V_b^{dc}`: voltage in kV.

Parameter Requirements for Voltage Rating
----------------------------------------------------
If a model is connected to an AC Bus or a DC Node, namely, ``bus``, ``bus1``, ``node``, or ``node1`` exist in
its parameter, it must provide the corresponding parameter, ``Vn``, ``Vn1``, ``Vdcn`` or ``Vdcn1``, for rated
voltages.

Controllers not connected to Bus or Node will have its rated voltages omitted and thus ``Vb = Vn = 1``.
In fact, controllers not directly connected to the network shall use per unit for voltage and current parameters
. Controllers (such as a turine governor) may inherit rated power from controlled models and thus power parameters
will be converted consistently.

Model and ModelData Classes
======================================

The `ModelData` class provides structure and methods for storing
power system data incrementally.

The `Model` class provides functions needed for defining
variables and equations.

OrderedDict of instances
-------------------------

Variables:
Variables has the following attributes in common:

*a*
  variable address
*v*
  variable value
*e*
  the corresponding equation value
*e_symbolic*
  the string/symbolic representation of the equation
*e_numeric*
  the callable to update equation value
*e_lambdify*
  the generated callable to update equation value

ExtVar:

External variables has the additional method:

*link_external()*
  linking to external variable

The following variable containers exist:

*states*
  for differential variables
*algebs*
  for algebraic variables
*calcs*
  for calculated variables
*vars_ext*
  for external variables

Parameters:

ParamBase hold the following attributes:

*property*
  for a dictionary of properties for data requirements
*v*
  for a list/array of values from input
*get_name()*
  returns a list only containing its name

NumParam holds the following additional attributes:

*pu_coeff*
  for coefficients for per-unit conversion
*vin*
  for a copy of the input variables
*params*
  for internal parameters
*params_ext*
  for external parameters

ExtParam holds the additional methods:

*link_external*
  for linking external parameter data

Service Constants:

*services*
  for service constants

Limiters:

Limiters are used to add limits to algebraic or state variables.
Limiters need be provided with a variable and its limits.

*limiters*
  for limiters

Blocks:

Blocks are collections of variables and the corresponding equations.
Blocks can be instantiated as model attributes. The instantiation of blocks
will add the corresponding variables and equations to the parent class.
An example block is the PIController.

*blocks*
  for general blocks


Sympify and Lambdify of Equations
====================================

Each variable provide two attributes for providing symbolic and
 numerical equations, respectively.
The symbolic equation is provided as a string in the variable's
 ``e_symbolic`` attribute.
The numerical equation is provided as a function call in the
 variable's ``e_numeric`` attribute.

The `convert_equation` function will convert the equation
defined in ``e_symbolic`` to a lambda function
and store it in the variable's `e_lambdify` attribute.
The conversion will store symbolic equations
in as a matrix in the ``g_syms_matrix``, ``f_syms_matrix``
and ``c_syms_matrix`` attributes, which will be used
for obtaining the Jacobian function calls.

The `convert_jacobian` function lambdifies the jacobians of
the equations, namely, the partial derivative of
``g_syms_matrix`` with respect to all the variable symbols,
``vars_syms``. The row indices (equation address),
column indices (variable address) and the lambdified derivative
functions will be stored in triplets, namely,

    (equation index, value index, lambdified function).

If the derivative is a constant, the triplet will be stored
in the corresponding list ending with a ``c``.
For example, the derivative of `df/df` will be stored in
``_fxc`` for constant derivative, and ``_fx``
for variable derivative.

A call to ``Model.get_sparse_pattern()`` will be made
to collect the rows and columns that contain a non-zero
element. The indices for ``df/fx``, for example,
will be stored in attributes ``Model.ifx``,
``model.jfx``.

Filling in the jacobian matrices involves calling
``Model.j_const_call()`` and ``Model.j_variable_call()``.
These two functions will iterate over the triplets in ``_fxc``
and ``_fx`` and directly modify the sparse matrix
``Model.system.dae.fx``. ``spmatrix.ipadd`` will be used
if available. Otherwise, it will a for loop and
in-place add.

Custom Numerical Equations
==========================
There are cases the user prefer or have to use numerical
functions, namely, Python functions, to update Equations and
Jacobians. To provide a numerical function call for equations,
the use needs to define a member function in the hosting
model. This function should update the equation value attribute,
``VarBase.e`` and return None. Then, this function should be
assigned to the ``e_numeric`` attribute of the corresponding
variable.

NEW: The ``e_numeric`` should take arguments of inputs in
its signature. For example::

  @static_method
  def _update_q(u, q, **kwargs):
      return u * q
