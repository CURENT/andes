.. _modeling:

**********************
Modeling
**********************

System
======================================


Models
======================================

Parameter Requirements for Voltage Rating
----------------------------------------------------
If a model is connected to an AC Bus or a DC Node, namely, ``bus``, ``bus1``, ``node``, or ``node1`` exist in
its parameter, it must provide the corresponding parameter, ``Vn``, ``Vn1``, ``Vdcn`` or ``Vdcn1``, for rated
voltages.

Controllers not connected to Bus or Node will have its rated voltages omitted and thus ``Vb = Vn = 1``.
In fact, controllers not directly connected to the network shall use per unit for voltage and current parameters
. Controllers (such as a turine governor) may inherit rated power from controlled models and thus power parameters
will be converted consistently.

..
    Atoms
    ANDES defines several types of atoms for building DAE models, including parameters, DAE variables,
    and service variables. Atoms can be used to build models and libraries, combined with discrete
    components and blocks.


Parameters
==============================
Parameters, in the scope of atoms, are data provided to equations. Parameters are usually read from input data
files and pre-processed before numerical simulation.

The base class for parameters in ANDES is ``BaseParam``, which defines interfaces for adding values and
checking the number of values. ``BaseParam`` has its values stored in a plain list, the member attribute ``v``.
Subclasses such as ``NumParam`` stores values using a NumPy ndarray. An overview of supported parameters is
given in the table below.

+---------------+----------------------------------------------------------------------------+
|  Subclasses   |     Description                                                            |
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


Variables
==============================
DAE Variables, or variables for short, are unknowns to be solved using numerical or analytical methods.
A variable stores values, equation values, and addresses in the DAE array. The base class for variables is
``VarBase``. In this subsection, ``VarBase`` is used to represent any subclass of ``VarBase`` list in the table
below.

+-----------+---------------------------------------------------------------------------------------+
|   Class   |                                      Description                                      |
+===========+=======================================================================================+
|  State    | A state variable and an associated differential equation :math:`\dot{x} = \textbf{f}` |
+-----------+---------------------------------------------------------------------------------------+
|  Algeb    | An algebraic variable and an associated algebraic equation :math:`0 = \textbf{g}`     |
+-----------+---------------------------------------------------------------------------------------+
|  ExtState | An external state variable and part of the differential equation (uncommon)           |
+-----------+---------------------------------------------------------------------------------------+
|  ExtAlgeb | An external algebraic variable and part of the algebraic equation                     |
+-----------+---------------------------------------------------------------------------------------+

``VarBase`` has two types: the differential variable type ``State`` and the algebraic variable type ``Algeb``.
State variables are described by differential equations, whereas algebraic variables are described by
algebraic equations. State variables can only change continuously, while algebraic variables
can be discontinuous.

Based on the model the variable is defined, variables can be internal or external. Most variables are internal
and only appear in equations in the same model. Some models have "public" variables that can be accessed by other
models. For example, a ``Bus`` defines ``v`` for the voltage magnitude.
Each device attached to a particular bus needs to access the value and impose the reactive power injection.
It can be done with ``ExtAlgeb`` or ``ExtState``, which links with an existing variable from a model or a group.

Variable, Equation and Address
------------------------------------------------
Subclasses of ``VarBase`` are value providers and equation providers.
Each ``VarBase`` has member attributes ``v`` and ``e`` for variable values and equation values, respectively.
The initial value of ``v`` is set by the initialization routine, and the initial value of ``e`` is set to zero.
In the process of power flow calculation or time domain simulation, ``v`` is not directly modifiable by models
but rather updated after solving non-linear equations. ``e`` is updated by the models and summed up before
solving equations.

Each ``VarBase`` also stores addresses of this variable, for all devices, in its member attribute ``a``. The
addresses are *0-based* indices into the numerical DAE array, ``f`` or ``g``, based on the variable type. For
example, ``Bus`` has ``a = Algeb()`` as the voltage phase angle variable. For a 5-bus system, ``Bus.a.a`` stores
the addresses of the ``a`` variable for all the five ``Bus`` devices. Conventionally, ``Bus.a.a`` will be
assigned ``np.array([0, 1, 2, 3, 4])``.

Value and Equation Strings
----------------------------------------
The most important feature of the symbolic framework is allowing to define equations using strings.
There are three types of strings for a variable, stored in the following member attributes, respectively:

- ``v_str``: equation string for **explicit** initialization in the form of ``v = v_str(x, y)``.
- ``v_iter``: equation string for **implicit** initialization in the form of ``v_iter(x, y) = 0``
- ``e_str``: equation string for (full or part of) the differential or algebraic equation.

The difference between ``v_str`` and ``v_iter`` should be clearly noted. ``v_str`` evaluates directly into the
initial value, while all ``v_iter`` equations are solved numerically using the Newton-Krylov iterative method.

Values Between DAE and Models
----------------------------------------
ANDES adopts a decentralized architecture which provides each model a copy of variable values before equation
evaluation. This architecture allows to parallelize the equation evaluation (in theory, or in practice if one
works round the Python GIL). However, this architecture requires a coherent protocol for updating the DAE arrays
and the ``VarBase`` arrays. More specifically, how the variable and equations values from model ``VarBase``
should be summed up or forcefully set at the DAE arrays needs to be defined.

The protocol is relevant when a model defines subclasses of ``VarBase`` that are supposed to be "public".
Other models share this variable with ``ExtAlgeb`` or ``ExtState``.
By default, all ``v`` and ``e`` at the same address are summed up.
This is the mose common case, such as a Bus connected by multiple devices: power injections from
devices should be summed up.

In addition, ``VarBase`` provides two flags, ``v_setter`` and ``e_setter``, for cases when one ``VarBase``
needs to overwrite the variable or equation values.

Flags for Value Overwriting
----------------------------------------
``VarBase`` have special flags for handling value initialization and equation values.
This is only relevant for public or external variables.
The ``v_setter`` is used to indicate whether a particular ``VarBase`` instance sets the initial value.
The ``e_setter`` flag indicates whether the equation associated with a ``VarBase`` sets the equation value.

The ``v_setter`` flag is checked when collecting data from models to the numerical DAE array. If
``v_setter is False``, variable values of the same address will be added.
If one of the variable or external variable has ``v_setter is True``, it will, at the end, set the values in the
DAE array to its value. Only one ``VarBase`` of the same address is allowed to have ``v_setter == True``.

The ``v_setter`` Example
----------------------------------------
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
                      v_str='v0',
                      v_setter=True)

where an ``ExtAlgeb`` is defined to access ``Bus.v`` using indexer ``self.bus``. The ``v_str`` line sets the
initial value to ``v0``. In the variable initialization phase for ``PV``, ``PV.v.v`` is set to ``v0``.

During the value collection into ``DAE.y`` by the ``System`` class, ``PV.v``, as a final ``v_setter``, will
overwrite the voltage magnitude for Bus devices with the indices provided in ``PV.bus``.

Services
======================================
Services are helper variables outside the DAE variable list. Services are most often used for storing intermediate
constants but can be used for special operations to work around restrictions in the symbolic framework.
Services are value providers, meaning each service has an attribute ``v`` for storing service values. The
base class of services is ``BaseService``, and the supported services are listd as follows.

+------------------+-----------------------------------------------------------------+
|      Class       |                           Description                           |
+==================+=================================================================+
|  ConstService    | Internal service for constant values.                           |
+------------------+-----------------------------------------------------------------+
|  ExtService      | External service for retrieving values from value providers.    |
+------------------+-----------------------------------------------------------------+
|  ReducerService  | The service type for reducing linear 2-D arrays into 1-D arrays |
+------------------+-----------------------------------------------------------------+
|  RepeaterService | The service type for repeating 1-D arrays to linear 2-D arrays  |
+------------------+-----------------------------------------------------------------+

``ConstService``
----------------------------------------
The most commonly used service is ``ConstService``.  It is used to store an array of constants, whose value is
evaluated from a provided symbolic string. They are only evaluated once in the model initialization phase, ahead
of variable initialization. ``ConstService`` comes handy when one wants to calculate intermediate constants from
parameters.

For example, a turbine governor has a ``NumParam`` ``R`` for the
droop. ``ConstService`` allows to calculate the inverse of the droop, the gain, and use it in equations. The
snippet from a turbine governor's ``__init__`` may look like ::

    self.R = NumParam()
    self.G = ConstService(v_str='u/R')

where ``u`` is the online status parameter. The model can thus use ``G`` in subsequent variable or equation
strings.

For more details, see the API doc: :py:mod:`andes.core.service.ConstService`

``ExtService``
----------------------------------------
Service constants whose value is retrieved from an external model or group. Using ``ExtService`` is
similar to using external variables. The values of ``ExtService`` will be retrieved once during the
initialization phase before ``ConstService`` evaluation.

For example, a synchronous generator needs to retrieve the ``p`` and ``q`` values from static generators
for initialization. ``ExtService`` is used for this purpose. In the ``__init__`` of a synchronous generator
model, one can define the following to retrieve ``StaticGen.p`` as ``p0``::

        self.p0 = ExtService(src='p',
                             model='StaticGen',
                             indexer=self.gen,
                             tex_name='P_0')

For more details, see the API doc: :py:mod:`andes.core.service.ExtService`

``ReducerService`` and ``RepeaterService``
-------------------------------------------
``ReducerService`` is a helper Service type which reduces a linearly stored 2-D ExtParam into 1-D Service.
``RepeaterService`` is a helper Service type which repeats a 1-D value into linearly stored 2-D value based on the
shape from a RefParam.

Both types are for advanced users. For more details and examples, please refer to the API documentation:

:py:mod:`andes.core.service.ReducerService`

:py:mod:`andes.core.service.RepeaterService`


Discrete
======================================


Blocks
======================================


Example: GENROU
======================================



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
