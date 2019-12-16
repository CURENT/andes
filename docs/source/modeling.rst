.. _modeling:

**********************
Modeling
**********************

System
======================================

Overview
----------------------------------------
:py:mod:`andes.System` is the top-level class for organizing and orchestrating a power system model. The
System class contains models and routines for modeling and simulation. ``System`` provides methods for
automatically importing groups, models, and routines at ``System`` creation.

``Systems`` contains a few special ``OrderedDict`` member attributes for housekeeping. These attributes include
``models``, ``groups``, ``programs`` and ``calls`` for loaded models, groups, program routines, and numerical
function calls, respectively. In these dictionaries, the keys are name strings and the values are the
corresponding instances.

Dynamic Imports
````````````````````````````````````````
Groups, models and routine programs are dynamically imported at the creation of a ``System`` instance. In
detail, *all classes* defined in ``andes.devices.group`` are imported and instantiated.
For models and groups, only the classes defined in the corresponding ``__init__.py`` files are imported and
instantiated in the order of definition.

See for details:

:py:mod:`andes.system.System._group_import()` : group import

:py:mod:`andes.system.System._model_import()` : model import

:py:mod:`andes.system.System._routine_import()` : routine import


Symbolic-to-Numeric Preparation
````````````````````````````````````````
Before the first use, all symbolic equations need to be generated into numerical function calls for accelerating
the numerical simulation. Since the symbolic to numeric generation is slow, these numerical
function calls (Python Callables), once generated, can be serialized into a file to speed up future. When models
are modified (such as adding new models or changing equation strings), the generation function needs to be
executed again for code consistency.

In the first use of ANDES in an interactive environment, one would do ::

    import andes
    sys = andes.System()
    sys.prepare()

It may take several seconds to a few minutes to finish the preparation.

The symbolic-to-numeric generation is independent of test systems and needs to happen before a test system is
loaded. In other words, any symbolic processing for particular test systems must not be included in
``System.prepare()``.

The package used for serializing/de-serializing numerical calls is ``dill``. The serialized file will be named
``calls.pkl`` and placed under ``<HomeDir>/.andes/``. As a note, the ``dill_calls()`` method has set the flag
``dill.settings['recurse'] = True`` to ensure a successful recursive serialization.

If no change is made to models, the call to ``prepare()`` afterwards can be replaced with ``undill_calls()``,
which is fast to execute.

See for details:

:py:mod:`andes.system.System.prepare()` : symbolic-to-numerical preparation

:py:mod:`andes.system.System.undill_calls()` : un-dill numerical calls

Numerical Functions
----------------------------------------

DAE Arrays and Sparse Matrices
````````````````````````````````````````
``System`` contains an instance of the numerical DAE class, ``System.dae``, for storing the numerical values of
variables, equations and first order derivatives (Jacobian matrices). Variable values and equation values are
stored in ``np.ndarray``, while Jacobians are stored in ``CVXOPT.spmatrix``. Defined arrays and descriptions are
as follows:

+-----------+---------------------------------------------+
| DAE Array |                 Description                 |
+===========+=============================================+
|  x        | Array for state variable values             |
+-----------+---------------------------------------------+
|  y        | Array for algebraic variable values.        |
+-----------+---------------------------------------------+
|  f        | Array for differential equation derivatives |
+-----------+---------------------------------------------+
|  g        | Array for algebraic equation mismatches     |
+-----------+---------------------------------------------+

Since the system of equations for power system simulation is determined, the number of equations has to equal
to the number of variables. In other words, ``x`` and ``f`` has the same length (stored in ``DAE.n``), and so do
``y`` and ``g`` (stored in ``DAE.m``).


The derivatives of ``f`` and ``g`` with respect to ``x`` and ``y`` are stored in four sparse matrices: ``fx``,
``fy``, ``gx`` and ``gy``, where the first letter is the equation name, and the second letter is the variable name.

Note that DAE does not store the original variable at a particular address. Conversely, the addresses of a
variable is stored in the variable instance. See Subsection Variables for more details.

Model and DAE Values
````````````````````````````````````````
ANDES uses a decentralized architecture between models and DAE value arrays. In this architecture, variables are
initialized and equations are evaluated inside each model. Since the equation system is solved simultaneously,
``System`` provides methods for collecting initial values and equation values into ``DAE``, as well as copying
updated variables to each model.

The collection of values from models need to follow protocols to avoid conflicts.  Details
are given in the subsection Variables.

See for more details:

:py:mod:`andes.System.vars_to_dae` : model -> DAE (for variable values)

:py:mod:`andes.System.vars_to_models` : DAE -> model (for variable values)

:py:mod:`andes.System._e_to_dae` : model -> DAE (for equation values)


Model Functions
````````````````````````````````````````
``System`` functions as an orchestrator for calling shared member methods of models. These methods are defined
for initialization, equation update, Jacobian update, and discrete flags update.

+--------------------------------------+------------------------------------------+
|            System Method             |               Description                |
+======================================+==========================================+
|  :py:mod:`andes.System.initialize`   | Variable initialization                  |
+--------------------------------------+------------------------------------------+
|  :py:mod:`andes.System.f_update`     | Update differential equation             |
+--------------------------------------+------------------------------------------+
|  :py:mod:`andes.System.g_update`     | Update algebraic equation                |
+--------------------------------------+------------------------------------------+
|  :py:mod:`andes.System.j_update`     | Update values in the Jacobians           |
+--------------------------------------+------------------------------------------+
|  :py:mod:`andes.System.l_update_var` | Discrete flags update based on variables |
+--------------------------------------+------------------------------------------+
|  :py:mod:`andes.System.l_update_eq`  | Discrete flags update based on equations |
+--------------------------------------+------------------------------------------+

Sparse Matrix Patterns
````````````````````````````````````````
The largest overhead in building and solving nonlinear equations is the building of Jacobian matrices. This is
especially relevant when we use the implicit integration approach which algebraized the differential equations.
Given the unique data structure of power system models, the sparse matrices for Jacobians are built model by
model, incrementally.

There are two common approaches to incrementally build a sparse matrix. The first one is to use simple in-place
add on sparse matrices, such as doing ::

    self.fx += spmatrix(v, i, j, (n, n), 'd')

Although the implementation is simple, this involves creating and discarding temporary objects on the right hand
side and, even worse, changing the sparse pattern of ``self.fx``. The second approach is to store the rows,
columns and values in an array-like object and construct the Jacobians at the end. This approach is very
efficient but has one caveat: it does not allow accessing the sparse matrix while building.

ANDES uses a hybrid approach to avoid the change of sparse patterns by filling values into a known the sparse
matrix pattern. ``System`` collects the indices of rows and columns for each Jacobian matrix. Before the
in-place addition, ANDES builds a temporary zero-filled ``spmatrix`` in which Jacobian values are updated.
Since these in-place add operations are only modifying existing values, it not change the pattern and thus will
not incur value copying. In addition, updating sparse matrices can use the exact same code as the first approach.

Note that this approach still creates and discards temporary objects, it is feasible to write a C function which
takes three array-likes and modify the sparse matrices in place. This is feature to be developed, and our
prototype shows a promising speed up.

See for details:

:py:mod:`andes.System.store_sparse_patterns` : store sparse patterns from models

Configuration
----------------------------------------
Each model and routine program has a member attribute ``config`` for model-specific or routine-specific
configurations. ``System`` also stores ``config`` for system-specific configurations. In addition, ``System``
manages collecting all configs, saving in a config file, and loading the config file.

The collected configs can be written to an ``andes.rc`` config file in ``<HomeDir>/.andes`` using
``ConfigParser``. Saved config file can be loaded and populated *at system instance creation time*. Configs from
the config file takes precedence over default config values.

Again, configs from files is passed to model constructors during instantiation. If one needs to modify the
config for a run, it needs to be done before the ``System`` instantiation. Directly modifying ``Model.config``
may not take effect or have side effect in the current implementation.


See for details:

:py:mod:`andes.common.Config` : Config class

:py:mod:`andes.System.save_config` : Save config into ``<HomeDir>/andes.rc``

:py:mod:`andes.System.load_config` : load config from ``<HomeDir>/andes.rc``

:py:mod:`andes.System._model_import` : dynamic model instantiation with config as an argument


Models
======================================

Overview
----------------------------------------

Parameters from Inputs
----------------------------------------
``ModelData``

Parameter Requirements for Voltage Rating
```````````````````````````````````````````````
If a model is connected to an AC Bus or a DC Node, namely, ``bus``, ``bus1``, ``node``, or ``node1`` exist in
its parameter, it must provide the corresponding parameter, ``Vn``, ``Vn1``, ``Vdcn`` or ``Vdcn1``, for rated
voltages.

Controllers not connected to Bus or Node will have its rated voltages omitted and thus ``Vb = Vn = 1``.
In fact, controllers not directly connected to the network shall use per unit for voltage and current parameters
. Controllers (such as a turine governor) may inherit rated power from controlled models and thus power parameters
will be converted consistently.

Completing Symbolic Equations
----------------------------------------

The ``__setattr`` magic


``Model.cache``
````````````````````````````````````````

Additional Numerical Equations
----------------------------------------




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

Appendix: Modeling Capability for PSS/E models
============================================================

Generator Models

TODO: InputSwitch, Saturation Block

+---------+---------------------------------+--------------+----------------------------------------+
| Model   | Description                     | Supportable? | Comments                               |
+---------+---------------------------------+--------------+----------------------------------------+
| CBEST   |                                 | 1            |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| CDSMS1  |                                 | 0            | Voltage mode calculation               |
+---------+---------------------------------+--------------+----------------------------------------+
| CGEN1   | Third-order complex generator   | 1            | d-q axis circuits are provided         |
+---------+---------------------------------+--------------+----------------------------------------+
| CIMTR4  | Induction generator model       |              | No control schematic in PSS/E manual   |
+---------+---------------------------------+--------------+----------------------------------------+
| CIMTR4  | Induction generator model       |              | No control schematic in PSS/E manual   |
+---------+---------------------------------+--------------+----------------------------------------+
| CSMEST  | EPRI V-&I-source SMES device    | 1            |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| CSTATT  | Static Condenser (STATCOM)      | 1            | Conditional maximum current limit      |
+---------+---------------------------------+--------------+----------------------------------------+
| CSVGN1  | Static Shunt Compensator        | 1            |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| CSVGN3  | Static Shunt Compensator        | 1            | Conditional feed-forward loop          |
+---------+---------------------------------+--------------+----------------------------------------+
| CSVGN4  | Static Shunt Compensator        | 1            | Differs from CSVGN4 in voltage input   |
+---------+---------------------------------+--------------+----------------------------------------+
| CSVGN5  | Static Shunt Compensator        | 1            |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| CSVGN6  | Static Shunt Compensator        | 1            | Input param as a selector              |
+---------+---------------------------------+--------------+----------------------------------------+
| FRECHG  | Frequency Charger Model         |              | No schematic                           |
+---------+---------------------------------+--------------+----------------------------------------+
| GENCLS  | Constant Vf generator model     | 1            |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| GENDCO  | Round rotor gen with dc         | 1            |                                        |
|         | offset torque component         |              |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| GENROE  | Round rotor gen with            | 1            |                                        |
|         | exponential saturation          |              |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| GENROU  | Round rotor gen with            | 1            |                                        |
|         | quadratic saturation            |              |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| GENSAE  | Salient pole gen with           | 1            |                                        |
|         | exp sat on both axes            |              |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| GENSAL  | Salient pole gen with           | 1            |                                        |
|         | quad sat on d-axis              |              |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| GENTPJ1 | WECC type J gen model           | 1            | Saturation through inductances         |
+---------+---------------------------------+--------------+----------------------------------------+
| GENTRA  | Transient level generator model | 1            |                                        |
+---------+---------------------------------+--------------+----------------------------------------+
| PLBVFU1 | Model to play-in known voltage  | 1            | An interesting model for data playback |
|         | and/or frequency signal         |              |                                        |
+---------+---------------------------------+--------------+----------------------------------------+


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
