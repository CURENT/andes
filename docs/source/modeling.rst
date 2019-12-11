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

Atoms
==============================
ANDES defines several types of atoms for building DAE models, including parameters, DAE variables,
and service variables. Atoms can be used to build models and libraries, combined with discrete
components and blocks.

Parameters
------------------------------
Parameters, in the scope of atoms, are data provided to equations. Parameters are usually read from input data
files and pre-processed before numerical simulation.

The base class for parameters in ANDES is ``BaseParam``, which defines interfaces for adding values and
checking the number of values. ``BaseParam`` has its values stored in a plain list, the member attribute ``v``.
Subclasses such as ``NumParam`` stores values using a NumPy ndarray. An overview of supported parameters is
given in the table below.

+---------------+----------------------------------------------------------------------------+
|  Subclasses   |     Usage                                                                  |
+===============+============================================================================+
|  DataParam    | An alias of ``BaseParam``. Can be used for any non-numerical parameters.   |
+---------------+----------------------------------------------------------------------------+
|  NumParam     | The numerical parameter type. Used for all parameters in equations         |
+---------------+----------------------------------------------------------------------------+
|  IdxParam     | The parameter type for storing ``idx`` into other models                   |
+---------------+----------------------------------------------------------------------------+
|  ExtParam     | Externally defined parameter                                               |
+---------------+----------------------------------------------------------------------------+
|  TimerParam   | Parameter for storing the action time of events                            |
+---------------+----------------------------------------------------------------------------+
|  RefParam     | Parameter for collecting ``idx`` of referencing devices                    |
+---------------+----------------------------------------------------------------------------+


DAE Variables
----------------------------------------
DAE Variables, or variables for short, are unknowns to be solved using numerical or analytical methods.
A variable stores values, equation values, and addresses in the DAE array.

Variables fall into two categories: state (differential) variables ``State`` and algebraic variables ``Algeb``.
State variables are described by differential equations, whereas algebraic variables are described by
algebraic equations. As a result, state variables can only change continuously, while algebraic variables
can be discontinuous.

Based on the model the variable is defined, variables can be internal or external. Most variables are internal
and only appear in equations in the same model. Some models have "public" variables that can be accessed by other
models. For example, a ``Bus`` defines ``v`` for the voltage magnitude.
Each device attached to a particular bus needs to access the value and impose the reactive power injection.
It can be done with ``ExtAlgeb`` or``ExtState``, which links with an existing variable from a model or a group.

Variables have special flags for handling value initialization and equation values. This is only relevant for
public or external variables. The ``v_setter`` is used to indicate whether a particular variable instance sets
the initial value. The ``e_setter`` flag indicates whether a particular equation associated with a variable sets
the equation value.

The ``v_setter`` flag is checked when collecting data from models to the numerical DAE array. If
``v_setter == False``, variable values of the same address will be added.
If one of the variable or external variable has ``v_setter == True``, it will, at the end, set the values in the
DAE array to its value. Only one (external) variable of the same address is allowed to have ``v_setter == True``.

An Example of ``v_setter``
```````````````````````````````````
A Bus is allowed to default the initial voltage magnitude to 1 and the voltage phase angle to 0.
If a PV device is connected to a Bus device, the PV should be allowed to override the voltage initial value
with the voltage set point.

In ``Bus.__init__``, one has ::

    self.v = Algeb(v_str='1')

In ``PV.__init__``, one can use ::

    self.v0 = Param()
    self.bus = IdxParam(model='Bus')

    self.v = ExtAlgeb(src='v',
                      model='Bus',
                      indexer=self.bus,
                      v_init='v0',
                      v_setter=True)

where an ``ExtAlgeb`` is defined to access ``Bus.v`` using indexer ``self.bus``. The ``v_init`` line sets the
initial value to ``v0``. During variable initialization for each model, ``v0`` is set in ``PV.v.v``.

During the value collection into the numerical DAE array, PV will overwrite the voltage magnitude of Bus devices
with indexes in ``PV.bus``.

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

BaseParam hold the following attributes:

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
``BaseVar.e`` and return None. Then, this function should be
assigned to the ``e_numeric`` attribute of the corresponding
variable.

NEW: The ``e_numeric`` should take arguments of inputs in
its signature. For example::

  @static_method
  def _update_q(u, q, **kwargs):
      return u * q
