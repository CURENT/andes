.. _modeling:

********
Modeling
********
This chapter contains advanced topics on modeling and simulation and how they are implemented in ANDES.
It aims to provide an in-depth explanation of how the ANDES framework is set up for symbolic modeling and
numerical simulation. It also provides an example for interested users to implement customized DAE models.

System
======

Overview
--------
:py:mod:`andes.System` is the top-level class for organizing and orchestrating a power system model. The
System class contains models and routines for modeling and simulation. ``System`` provides methods for
automatically importing groups, models, and routines at ``System`` creation.

``Systems`` contains a few special ``OrderedDict`` member attributes for housekeeping. These attributes include
``models``, ``groups``, ``programs`` and ``calls`` for loaded models, groups, program routines, and numerical
function calls, respectively. In these dictionaries, the keys are name strings and the values are the
corresponding instances.

Dynamic Imports
```````````````
Groups, models and routine programs are dynamically imported at the creation of a ``System`` instance. In
detail, *all classes* defined in ``andes.models.group`` are imported and instantiated.
For models and groups, only the classes defined in the corresponding ``__init__.py`` files are imported and
instantiated in the order of definition.

Code Generation
```````````````
Before the first use, all symbolic equations need to be generated into numerical function calls for accelerating
the numerical simulation. Since the symbolic to numeric generation is slow, these numerical
function calls (Python Callables), once generated, can be serialized into a file to speed up future. When models
are modified (such as adding new models or changing equation strings), the generation function needs to be
executed again for code consistency.

In the first use of ANDES in an interactive environment, one would do ::

    import andes
    sys = andes.System()
    sys.prepare()

It may take several seconds to a few minutes to finish the code generation. This process is automatically invoked
for the first time ANDES is run command line. The preparation process can be manually triggered with
``andes --prepare``.

The symbolic-to-numeric code generation is independent of test systems and needs to happen before a test system is
loaded. In other words, any symbolic processing for particular test systems must not be included in
``System.prepare()``.

The package used for serializing/de-serializing numerical calls is ``dill``. The serialized file will be named
``calls.pkl`` and placed under ``<HomeDir>/.andes/``. As a note, the ``dill()`` method has set the flag
``dill.settings['recurse'] = True`` to ensure a successful recursive serialization.

If no change is made to models, the call to ``prepare()`` afterwards can be replaced with ``undill()``,
which is fast to execute.

See for details:

:py:mod:`andes.system.System.prepare()` : symbolic-to-numerical preparation

:py:mod:`andes.system.System.undill()` : un-dill numerical calls

Numerical Functions
-------------------

DAE Arrays and Sparse Matrices
``````````````````````````````
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
````````````````````
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
```````````````
``System`` functions as an orchestrator for calling shared member methods of models. These methods are defined
for initialization, equation update, Jacobian update, and discrete flags update.

+--------------------------------------+------------------------------------------+
|            System Method             |               Description                |
+======================================+==========================================+
|  :py:mod:`andes.System.init`         | Variable initialization                  |
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
``````````````````````
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
-------------
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
======
This section introduces the modeling of power system devices. The terminology "model" is used to describe the
mathematical representation of a type of device, such as synchronous generators and turine governors. The
terminology "device" is used to describe a particular instance of a model, for example, a specific generator.

To define a model in ANDES, two classes, ``ModelData`` and ``Model`` need to be utilized. Class ``ModelData`` is
used for defining parameters that will be provided from input files. It provides API for adding data from
devices and managing the data. Class ``Model`` is used for defining other non-input parameters, service
variables, and DAE variables. It provides API for converting symbolic equations, storing Jacobian patterns, and
updating equations.

Parameters from Inputs
----------------------
Class ``ModelData`` needs to be inherited to create the class holding the input parameters for a new model. The
recommended name for the derived class is the model name with ``Data``. In ``__init__`` of the derived class,
the input parameters can be defined. Note that two default parameters, ``u`` (connection status, ``NumParam``),
and ``name`` (device name, ``DataParam``) are defined in ``ModelBase``), and it will apply to all subclasses.

Refer to the Parameters subsection for available parameter types.

For example, if we need to build the ``PQData`` class (for static PQ load) with three parameters, ``Vn``, ``p0``
and ``q0``, we can use the following ::

    from andes.core.model import ModelData, Model
    from andes.core.param import IdxParam, NumParam, DataParam

    class PQData(ModelData):
        super().__init__()
        self.Vn = NumParam(default=110,
                           info="AC voltage rating",
                           unit='kV', non_zero=True,
                           tex_name=r'V_n')
        self.p0 = NumParam(default=0,
                           info='active power load in system base',
                           tex_name=r'p_0', unit='p.u.')
        self.q0 = NumParam(default=0,
                           info='reactive power load in system base',
                           tex_name=r'q_0', unit='p.u.')

In this example, all the three parameters are defined as ``NumParam``. In the full ``PQData`` class, other
types of parameters also exist. For example, to store the idx of ``Owner``, ``PQData`` has ::

        self.owner = IdxParam(model='Owner', info="owner idx")

``Model.cache``
```````````````
``ModelData`` uses a lightweight class ``Cache`` for caching its data as a dictionary or a pandas Dataframe.
Four attributes are defined for ``ModelData.cache``:

- ``dict``: all data in a dictionary with the parameter names as keys and ``v`` values as arrays.
- ``dict_in``: the same as ``dict`` except that the values are from ``v_in``, the original input
- ``df``: all data in a pandas DataFrame.
- ``df_in``: the same as ``df`` except that the values are from ``v_in``

Other attributes can be added, if necessary, by registering with ``cache.add_callback``. An argument-free
callback function needs to be provided. See the source code of ``ModelData`` for details.

Parameter Requirements for Voltage Rating
`````````````````````````````````````````
If a model is connected to an AC Bus or a DC Node, namely, ``bus``, ``bus1``, ``node``, or ``node1`` exist in
its parameter, it must provide the corresponding parameter, ``Vn``, ``Vn1``, ``Vdcn`` or ``Vdcn1``, for rated
voltages.

Controllers not connected to Bus or Node will have its rated voltages omitted and thus ``Vb = Vn = 1``.
In fact, controllers not directly connected to the network shall use per unit for voltage and current parameters
. Controllers (such as a turine governor) may inherit rated power from controlled models and thus power parameters
will be converted consistently.


Defining a DAE Model
--------------------
After subclassing ``ModelData``, ``Model`` needs to be derived to complete a DAE model. Subclasses of Model
defines DAE variables, service variables, and other types of parameters, in the constructor ``__init__``, to
complete a model.

Again, take the static PQ as an example, the subclass of ``Model``, ``PQ``, looks like ::


    class PQ(PQData, Model):
        def __init__(self, system=None, config=None):
            PQData.__init__(self)
            Model.__init__(self, system, config)

In this case, ``PQ`` is meant to be the final class, not to be further derived. It inherits from ``PQData``
and ``Model``, calls the constructors in the order of ``PQData`` and ``Model``. Note that if the derived class
or ``Model`` is meant to be further derived, it should only derive from ``Model`` and use a name ending with
``Base``. See ``GENBase`` in ``models/synchronous.py`` for example.

Next, in ``PQ.__init__``, the proper flags for the routines the model will participate needs to be set. ::

    self.flags.update({'pflow': True})

Currently, flags ``pflow`` and ``tds`` are supported. They are ``False`` by default, meaning the model is
neither used in power flow nor time-domain simulation. **A very common pitfall is forgetting to set the flag**.

Next, the group name can be provided. A group is a collection of models with common parameters and variables.
Devices idx of all models in the same group must be unique. To provide a group name, use ::

    self.group = 'StaticLoad'

The group name must be an existing class name in ``models/groups.py``. The model will be added to the specified
group and subject to variable and parameter policy by the group. Otherwise, the model will be placed in the
``Undefined`` group.

Next, additional configuration flags can be added. Configuration flags for models are load-time variables
specifying the behavior of a model. It can be exported to an ``andes.rc`` file and automatically loaded when
creating the ``System``. Configuration flags can be used in equation strings, as long as they are numerical
values. To add configuration flags, use ::

    self.config.add(OrderedDict((('pq2z', 1), )))

It is recommended to use ``OrderedDict``, although the syntax is a bit verbose. Note that booleans should be
provided in integers (1, or 0), since ``True`` or ``False`` is interpreted as strings when loaded from an ``rc``
file and will cause an error.

Next, it's time for variables and equations! The ``PQ`` class does not have internal variables itself. It uses
its ``bus`` attribute to fetch the corresponding ``a`` and ``v`` variables of buses. Equation wise, it imposes
an active power and a reactive power demand equation.

To define external variables from ``Bus``, use ::

        self.a = ExtAlgeb(model='Bus', src='a',
                          indexer=self.bus, tex_name=r'\theta')
        self.v = ExtAlgeb(model='Bus', src='v',
                          indexer=self.bus, tex_name=r'V')

Refer to details in subsection Variables for more details.

The simplest ``PQ`` model will impose constant P and Q, coded as ::

        self.a.e_str = "u * p"
        self.v.e_str = "u * q"

where the ``e_str`` attribute is the equation string attribute. ``u`` is the connectivity status. Any parameter,
config, service or variables can be used in equation strings.

The above example is overly simplified. Further, our ``PQ`` model wants a feature to switch itself to
a constant impedance if the voltage is out of the range ``(vmin, vmax)``. To implement this, we need to
introduce a discrete component called ``Limiter``, which yields three arrays of binary flags, ``zi``, ``zl``, and
``zu`` indicating in the range, below lower limit, and above upper limit, respectively.

First, create an attribute ``vcmp`` as a ``Limiter`` instance ::

        self.vcmp = Limiter(u=self.v, lower=self.vmin, upper=self.vmax,
                             enable=self.config.pq2z)

where ``self.config.pq2z`` is a flag to turn this feature on or off.After this line, we can use ``vcmp_zi``,
``vcmp_zl``, and ``vcmp_zu`` in equation strings. ::

        self.a.e_str = "u * (p0 * vcmp_zi + \
                             p0 * vcmp_zl * (v ** 2 / vmin ** 2) + \
                             p0 * vcmp_zu * (v ** 2 / vmax ** 2))"

        self.v.e_str = "u * (q0 * vcmp_zi + \
                             q0 * vcmp_zl * (v ** 2 / vmin ** 2) + \
                             q0 * vcmp_zu * (v ** 2 / vmax ** 2))"

The two equations above implements a piecewise power injection equation. It selects the original power demand
if within range, and uses the calculated power when out of range.

Finally, to let ANDES pick up the model, the model name needs to be added to ``models/__init__.py``. Follow the
examples in the ``OrderedDict``, where the key is the file name, and the value is the class name.

Dynamicity Under the Hood
-------------------------
The magic for automatic creation of variables are all hidden in ``Model.__setattr__``, and the code is
incredible simple. It sets the name, tex_name, and owner model of the attribute instance and, more importantly,
does the book keeping. In particular, when the attribute is a ``Block`` subclass, ``__setattr__`` captures the
exported instances, recirsively, and prepends the block name to exported ones. All these convenience owe to the
dynamic feature of Python.

During the equation generation phase, the symbols created by checking the book-keeping attributes, such as
``states`` and attributes in ``Model.cache``.

In the numerical evaluation phase, ``Model`` provides a method, ``get_inputs`` to collect the variable value
arrays in a dictionary, which can be effortlessly passed to numerical functions.

Commonly Used Attributes in Models
``````````````````````````````````
The following ``Model`` attributes are commonly used for debugging. If the attribute if an ``OrderedDict``, the
key is usually the attribute name, and the value is the instance.

- ``params`` and ``params_ext``, two ``OrderedDict`` for internal and extenal parameters, respectively.
- ``states`` and ``algebs``, two ``OrderedDict`` for state variables and algebraic variables, respectively.
- ``states_ext`` and ``algebs_ext``, two ``OrderedDict`` for external states and algebraics.
- ``discrete``, an ``OrderedDict`` for discrete components.
- ``blocks``, an ``OrderedDict`` for blocks.
- ``services``, an ``OrderedDict`` for services with ``v_str``.
- ``services_ext``, an ``OrderedDict`` for externally retrieved services.

Attributes in ``Model.cache``
`````````````````````````````
Attributes in ``Model.cache`` are additional book-keeping structures for variables, parameters and services. THe
following attributes are defined in ``Model.cache``.

- ``all_vars``: all the variables
- ``all_vars_names``, a list of all variable names
- ``all_params``, all parameters
- ``all_params_names``, a list of all parameter names
- ``algebs_and_ext``, an ``OrderedDict`` of internal and external algebraic variables
- ``states_and_ext``, an ``OrderedDict`` of internal and external differential variables
- ``services_and_ext``, an ``OrderedDict`` of internal and external service variables.
- ``vars_int``, an ``OrderedDict`` of all internal variables, states and then algebs
- ``vars_ext``, an ``OrderedDict`` of all external variables, states and then algebs

Equation Generation
-------------------
``Model`` handles the symbolic to numeric generation when called. The equation generation is a multi-step
process with symbol preparation, equation generation, Jacobian generation, initializer generation, and pretty
print generation.

The symbol preparation prepares ``OrderedDict`` of ``input_syms``, ``vars_syms`` and ``non_vars_syms``.
``input_syms`` contains all possible symbols in equations, including variables, parameters, discrete flags, and
config flags. ``input_syms`` has the same variables as what ``get_inputs()`` returns. Besides, ``vars_syms`` are
the variable-only symbols, which are useful when getting the Jacobian matrices. ``non_vars_syms`` contains the
symbols in ``input_syms`` but not in ``var_syms``.

Next, function ``generate_equation`` converts each DAE equation set to one numerical function calls and store
it in ``Model.calls``. The attributes for differential equation set and algebraic equation set are ``f_lambdify``
and ``g_lambdify``. Differently, service variables will be generated one by one and store in an ``OrderedDict``
in ``Model.calls.s_lambdify``.


Jacobian Storage
----------------

Abstract Jacobian Storage
`````````````````````````
Using the ``.jacobian`` method on ``sympy.Matrix``, the symbolic Jacobians can be easily obtains. The complexity
lies in the storage of the Jacobian elements. Observed that the Jacobian equation generation happens before any
system is loaded, thus only the variable indices in the variable array is available. For each non-zero item in each
Jacobian matrix, ANDES stores the equation index, variable index, and the Jacobian value (either a constant
number or a callable function returning an array).

Note that, again, a non-zero entry in a Jacobian matrix can be either a constant or an expression. For efficiency,
constant numbers and lambdified callables are stored separately. Constant numbers, therefore, can be loaded into
the sparse matrix pattern when a particular system is given.

The triplets, the equation (row) index, variable (column) index, and values (constant numbers or callable) are
stored in ``Model`` attributes with the name of ``_{i, j, v}{Jacobian Name}{c or None}``, where
``{i, j, v}`` is a single character for row, column or value, ``{Jacobian Name}`` is a two-character Jacobian
name chosen from ``fx, fy, gx, and gy``, and ``{c or None}`` is either character ``c`` or no character,
indicating whether it corresponds to the constants or non-constants in the Jacobian.

For example, the triplets for the
constants in Jacobian ``gy`` are stored in ``_igyc``, ``_jgyc``, and ``_vgyc``.

In terms of the non-constant entries in Jacobians, the callable functions are stored in the corresponding
``_v{Jacobian Name}`` array. Note the differences between, for example, ``_vgy`` an ``_vgyc``: ``_vgy`` is a
list of callables, while ``_vgyc`` is a list of constant numbers.

Concrete Jacobian Storage
`````````````````````````
When a specific system is loaded and the addresses are assigned to variables, the abstract Jacobian triplets,
more specifically, the rows and columns, are replaced with the array of addresses. The new addresses and values
will be stored in ``Model`` attributes with the names ``{i, j, v}{Jacobian Name}{c or None}``. Note that there
is no underscore for the concrete Jacobian triplets.

For example, if model ``PV`` has a list of variables ``[p, q, a, v]`` .
The equation associated with ``p`` is ``- u * p0``, and the equation associated with ``q`` is ``u * (v0 - v)``.
Therefore, the derivative of equation ``v0 - v`` over ``v`` is ``-u``. Note that ``u`` is unknown at generation
time, thus the value is NOT a constant and should to go ``vgy``.

The values in ``_igy``, ``_jgy`` and ``_vgy`` contains, respectively, ``1``, ``3``, and a lambda function which
returns ``-u``.

When a specific system is loaded, for example, a 5-bus system, the addresses for the ``q`` and ``v`` are ``[11,
13, 15``, and ``[5, 7, 9]``.
``PV.igy`` and ``PV.jgy`` will thus query the corresponding address list based on ``PV._igy`` and ``PV._jgy``
and store ``[11, 13, 15``, and ``[5, 7, 9]``.

Initialization
--------------
Value providers such as services and DAE variables need to be initialized. Services are initialized before
any DAE variable. Both Services and DAE Variables are initialized *sequentially* in the order of declaration.

Each Service, in addition to the standard ``v_str`` for symbolic initialization, provides a ``v_numeric`` hook
for specifying a custom function for initialization. Custom initialization functions for DAE variables, are
lumped in a single function in ``Model.v_numeric``.

ANDES has an *experimental* Newton-Krylov method based iterative initialization. All DAE variables with ``v_iter``
will be initialized using the iterative approach

Additional Numerical Equations
------------------------------
Addition numerical equations are allowed to complete the "hybrid symbolic-numeric" framework. Numerical function
calls are useful when the model DAE is non-standard or hard to be generalized. Since the
symbolic-to-numeric generation is an additional layer on top of the numerical simulation, it is fundamentally
the same as user-provided numerical function calls.

ANDES provides the following hook functions in each ``Model`` subclass for custom numerical functions:

- ``v_numeric``: custom initialization function
- ``s_numeric``: custom service value function
- ``g_numeric``: custom algebraic equations; update the ``e`` of the corresponding variable.
- ``f_numeric``: custom differential equations; update the ``e`` of the corresponding variable.
- ``j_numeric``: custom Jacobian equations; the function should append to ``_i``, ``_j`` and ``_v`` structures.

For most models, numerical function calls are unnecessary and not recommended as it increases code complexity.
However, when the data structure or the DAE are difficult to generalize in the symbolic framework, the numerical
equations can be used.

For interested readers, see the ``COI`` symbolic implementation which calculated the
center-of-inertia speed of generators. The ``COI`` could have been implemented numerically with for loops
instead of ``ReducerService``, ``RepeaterService`` and external variables.

..
    Atoms
    ANDES defines several types of atoms for building DAE models, including parameters, DAE variables,
    and service variables. Atoms can be used to build models and libraries, combined with discrete
    components and blocks.


Parameters
==========
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
=========
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
------------------------------
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
--------------------------
The most important feature of the symbolic framework is allowing to define equations using strings.
There are three types of strings for a variable, stored in the following member attributes, respectively:

- ``v_str``: equation string for **explicit** initialization in the form of ``v = v_str(x, y)``.
- ``v_iter``: equation string for **implicit** initialization in the form of ``v_iter(x, y) = 0``
- ``e_str``: equation string for (full or part of) the differential or algebraic equation.

The difference between ``v_str`` and ``v_iter`` should be clearly noted. ``v_str`` evaluates directly into the
initial value, while all ``v_iter`` equations are solved numerically using the Newton-Krylov iterative method.

Values Between DAE and Models
-----------------------------
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
---------------------------
``VarBase`` have special flags for handling value initialization and equation values.
This is only relevant for public or external variables.
The ``v_setter`` is used to indicate whether a particular ``VarBase`` instance sets the initial value.
The ``e_setter`` flag indicates whether the equation associated with a ``VarBase`` sets the equation value.

The ``v_setter`` flag is checked when collecting data from models to the numerical DAE array. If
``v_setter is False``, variable values of the same address will be added.
If one of the variable or external variable has ``v_setter is True``, it will, at the end, set the values in the
DAE array to its value. Only one ``VarBase`` of the same address is allowed to have ``v_setter == True``.

The ``v_setter`` Example
------------------------
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
========
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
----------------
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
--------------
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
========
The discrete component library contains a special type of block for modeling the discontinuity in power system
devices. Such continuities can be device-level physical constraints or algorithmic limits imposed on controllers.

The base class for discrete components is :py:mod:`andes.core.discrete.Discrete`. ANDES includes the following
types of discrete components

+--------------------+---------------------------------------------------------+
|   Discrete Class   |                       Description                       |
+====================+=========================================================+
|  Limiter           | Basic limiter with upper and lower bound                |
+--------------------+---------------------------------------------------------+
|  SortedLimiter     | Limiter with the top N values flagged                   |
+--------------------+---------------------------------------------------------+
|  HardLimiter       | Hard limiter on algebraic variables                     |
+--------------------+---------------------------------------------------------+
|  WindupLimiter     | Windup limiter on state variables                       |
+--------------------+---------------------------------------------------------+
|  AntiWindupLimiter | Non-windup limiter on state variables                   |
+--------------------+---------------------------------------------------------+
|  DeadBand          | Deadband with return flags                              |
+--------------------+---------------------------------------------------------+
|  Selector          | Selector with values matching the output of the         |
|                    | selection function                                      |
+--------------------+---------------------------------------------------------+
|  Switcher          | Input switcher with one array of flag for each input    |
|                    | option                                                  |
+--------------------+---------------------------------------------------------+

The uniqueness of discrete components is how it works. Discrete components take inputs, criteria, and exports a
set of flags with the component-defined meanings. These exported flags can be used in algebraic or
differential equations to build piece-wise equations.

For example, ``Limiter`` takes a v-provider as input, two v-providers as the upper and the lower bound. It
yields three flags: ``zi`` (within bound), ``zl`` (below lower bound), and ``zu`` (above upper bound). See the
code example in ``models/pv.py`` for an example voltage-based PQ-to-Z conversion. See the API references for
more examples on all types of discrete components.

It is important to note when the flags are updated. Discrete subclasses can use
four methods to check and update the value and equations. Among these methods, ``check_var`` and ``set_var`` are
called *before* equation evaluation, and ``check_eq`` and ``set_eq`` are called *after* equation update. In the
current implementation, ``check_var`` updates flags for variable-based discrete components (such as ``Limiter``)
. ``check_eq`` updates flags for equation-involved discrete componets (such as ``AntiWindupLimiter``).
``set_var`` is currently not used. It is recommended not to use ``set_var`` and, instead, use the flags in
equations to maintain consistency between equations and Jacobians.


Blocks
======
The block library contains commonly used transfer functions and nonlinear functions. Variables and equations are
pre-defined for blocks to be used as lego pieces for scripting DAE models. The base class for blocks is
:py:mod:`andes.core.block.Block`.

The supported blocks include ``Lag``, ``LeadLag``, ``Washout``, ``LeadLagLimit``, ``PIController``. In addition,
the base class for piece-wise nonlinear functions, ``PieceWise`` is provided. ``PieceWise`` is used for
implementing the quadratic saturation function ``MagneticQuadSat`` and exponential saturation function
``MagneticExpSat``.

All variables in a block must be defined as attributes in the constructor, just like variable definition in
models. The difference is that the variables are "exported" from a block to the capturing model. All exported
variables need to placed in a dictionary, ``self.vars`` at the end of the block constructor.


Blocks can be nexted as advanced usage. See the API documentation for more details.

Examples
========

TGOV1
-----
The TGOV1 turbine governor model is used as a practical example using the library.

This model is composed of a lead-lag transfer function and a first-order lag transfer function
with an anti-windup limiter, which are sufficiently complex for demonstration.
The corresponding differential equations and algebraic equations are given below.

.. math::

    \left[
    \begin{matrix}
    \dot{x}_{LG} \\
    \dot{x}_{LL}
    \end{matrix}
    \right]
    =
    \left[
    \begin{matrix}z_{i,lim}^{LG} \left(P_{d} - x_{LG}\right) / {T_1}
    \\
    \left(x_{LG} - x_{LL}\right) / T_3
    \end{matrix}
    \right]

    \left[
    \begin{matrix}
    0 \\
    0 \\
    0 \\
    0 \\
    0 \\
    0
    \end{matrix}
    \right]
    =
    \left[
    \begin{matrix}
    (1 - \omega) - \omega_{d} \\
    R \times \tau_{m0} - P_{ref} \\
    \left(P_{ref} + \omega_{d}\right)/R - P_{d}\\
    D_{t} \omega_{d} + y_{LL}  - P_{OUT}\\
    \frac{T_2}{T_3} \left(x_{LG} - x_{LL}\right) + x_{LL} - y_{LL}\\
    u \left(P_{OUT} - \tau_{m0}\right)
    \end{matrix}
    \right]

where *LG* and *LL* denote the lag block and the lead-lag block, :math:`\dot{x}_{LG}` and :math:`\dot{x}_{LL}`
are the internal states, :math:`y_{LL}` is the lead-lag output, :math:`\omega` the generator speed,
:math:`\omega_d` the generator under-speed, :math:`P_d` the droop output, :math:`\tau_{m0}` the steady-state
torque input, and :math:`P_{OUT}` the turbine output that will be summed at the generator.

The code for the above model is demonstrated as follows. The complete code can be found in
``andes/models/governor.py``. ::

    def __init__(self):
      # 1. Declare parameters from case file inputs.
      self.R = NumParam(info='Turbine governor droop',
                        non_zero=True, ipower=True)
      # Other parameters are omitted.

      # 2. Declare external variables from generators.
      self.omega = ExtState(src='omega',
                     model='SynGen',
                     indexer=self.syn,
                     info='Generator speed')
      self.tm = ExtAlgeb(src='tm',
                  model='SynGen',
                  indexer=self.syn,
                  e_str='u*(pout-tm0)',
                  info='Generator torque input')

      # 3. Declare initial values from generators.
      self.tm0 = ExtService(src='tm',
                   model='SynGen',
                   indexer=self.syn,
                   info='Initial torque input')

      # 4. Declare variables and equations.
      self.pref = Algeb(info='Reference power input',
                    v_str='tm0*R',
                    e_str='tm0*R-pref')
      self.wd = Algeb(info='Generator under speed',
                  e_str='(1-omega)-wd')
      self.pd = Algeb(info='Droop output',
                  v_str='tm0',
                  e_str='(wd+pref)/R-pd')
      self.LG_x = State(info='State in the lag TF',
                    v_str='pd',
                    e_str='LG_lim_zi*(pd-LG_x)/T1')
      self.LG_lim = AntiWindup(u=self.LG_x,
                      lower=self.VMIN,
                      upper=self.VMAX)
      self.LL_x = State(info='State in the lead-lag TF',
                    v_str='LG_x',
                    e_str='(LG_x-LL_x)/T3')
      self.LL_y = Algeb(info='Lead-lag Output',
                    v_str='LG_x',
                    e_str='T2/T3*(LG_x-LL_x)+LL_x-LL_y')
      self.pout = Algeb(info='Turbine output power',
                    v_str='tm0',
                    e_str='(LL_y+Dt*wd)-pout')

