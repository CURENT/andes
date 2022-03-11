Models
======
This section introduces the modeling of power system devices. The terminology "model" is used to describe the
mathematical representation of a *type* of device, such as synchronous generators or turbine governors. The
terminology "device" is used to describe a particular instance of a model, for example, a specific generator.

To define a model in ANDES, two classes, ``ModelData`` and ``Model`` need to be utilized. Class ``ModelData`` is
used for defining parameters that will be provided from input files. It provides API for adding data from
devices and managing the data.
Class ``Model`` is used for defining other non-input parameters, service
variables, and DAE variables. It provides API for converting symbolic equations, storing Jacobian patterns, and
updating equations.

The following classes are related to models:


.. currentmodule:: andes.core.model
.. autosummary::
      :recursive:
      :toctree: _generated

      ModelData
      Model
      ModelCache
      ModelCall


Cache
`````
`ModelData` uses a lightweight class :py:class:`andes.core.model.ModelCache`
for caching its data as a dictionary
or a pandas DataFrame. Four attributes are defined in `ModelData.cache`:

- `dict`: all data in a dictionary with the parameter names as keys and `v` values as arrays.
- `dict_in`: the same as `dict` except that the values are from `v_in`, the original input.
- `df`: all data in a pandas DataFrame.
- `df_in`: the same as `df` except that the values are from `v_in`.

Other attributes can be added by registering with `cache.add_callback`.

.. autofunction:: andes.core.model.ModelCache.add_callback
    :noindex:

Define Voltage Ratings
``````````````````````
If a model is connected to an AC Bus or a DC Node, namely, if ``bus``, ``bus1``, ``node`` or ``node1`` exists
as parameter, it must provide the corresponding parameter, ``Vn``, ``Vn1``, ``Vdcn`` or ``Vdcn1``, for rated
voltages.

Controllers not connected to Bus or Node will have its rated voltages omitted and thus ``Vb = Vn = 1``, unless
one uses :py:class:`andes.core.param.ExtParam` to retrieve the bus/node values.

As a rule of thumb, controllers not directly connected to the network shall use system-base per unit for voltage
and current parameters.
Controllers (such as a turbine governor) may inherit rated power from controlled models and thus power parameters
will be converted consistently.


Define a DAE Model
--------------------
.. autoclass:: andes.core.model.Model
    :noindex:

Dynamicity Under the Hood
-------------------------
The magic for automatic creation of variables are all hidden in :py:func:`andes.core.model.Model.__setattr__`,
and the code is incredible simple.
It sets the name, tex_name, and owner model of the attribute instance and, more importantly,
does the book keeping.
In particular, when the attribute is a :py:class:`andes.core.block.Block` subclass, ``__setattr__`` captures the
exported instances, recursively, and prepends the block name to exported ones.
All these convenience owe to the dynamic feature of Python.

During the code generation phase, the symbols are created by checking the book-keeping attributes, such as
`states`, `algebs`, and attributes in `Model.cache`.

In the numerical evaluation phase, `Model` provides a method, :py:func:`andes.core.model.get_inputs`, to
collect the variable value arrays in a dictionary, which can be effortlessly passed as arguments to numerical
functions.

Commonly Used Attributes in Models
``````````````````````````````````
The following ``Model`` attributes are commonly used for debugging.
If the attribute is an `OrderedDict`, the keys are attribute names in str, and corresponding values are the
instances.

- ``params`` and ``params_ext``, two `OrderedDict` for internal (both numerical and non-numerical) and external
  parameters, respectively.
- ``num_params`` for numerical parameters, both internal and external.
- ``states`` and ``algebs``, two ``OrderedDict`` for state variables and algebraic variables, respectively.
- ``states_ext`` and ``algebs_ext``, two ``OrderedDict`` for external states and algebraics.
- ``discrete``, an `OrderedDict` for discrete components.
- ``blocks``, an `OrderedDict` for blocks.
- ``services``, an `OrderedDict` for services with ``v_str``.
- ``services_ext``, an `OrderedDict` for externally retrieved services.

Attributes in `Model.cache`
```````````````````````````
Attributes in `Model.cache` are additional book-keeping structures for variables, parameters and services.
The following attributes are defined.

- ``all_vars``: all the variables.
- ``all_vars_names``, a list of all variable names.
- ``all_params``, all parameters.
- ``all_params_names``, a list of all parameter names.
- ``algebs_and_ext``, an `OrderedDict` of internal and external algebraic variables.
- ``states_and_ext``, an `OrderedDict` of internal and external differential variables.
- ``services_and_ext``, an `OrderedDict` of internal and external service variables.
- ``vars_int``, an `OrderedDict` of all internal variables, states and then algebs.
- ``vars_ext``, an `OrderedDict` of all external variables, states and then algebs.

Equation Generation
-------------------
``Model.syms``, an instance of ``SymProcessor``, handles the symbolic to numeric generation when called. The
equation generation is a multi-step process with symbol preparation, equation generation, Jacobian generation,
initializer generation, and pretty print generation.

.. autoclass:: andes.core.model.SymProcessor
    :members: generate_symbols, generate_equations, generate_jacobians, generate_init
    :noindex:

Next, function ``generate_equation`` converts each DAE equation set to one numerical function calls and store
it in ``Model.calls``. The attributes for differential equation set and algebraic equation set are ``f``
and ``g``. Differently, service variables will be generated one by one and store in an ``OrderedDict``
in ``Model.calls.s``.


Jacobian Storage
----------------

Abstract Jacobian Storage
`````````````````````````
Using the ``.jacobian`` method on ``sympy.Matrix``, the symbolic Jacobians can be easily obtained. The complexity
lies in the storage of the Jacobian elements. Observed that the Jacobian equation generation happens before any
system is loaded, thus only the variable indices in the variable array is available. For each non-zero item in each
Jacobian matrix, ANDES stores the equation index, variable index, and the Jacobian value (either a constant
number or a callable function returning an array).

Note that, again, a non-zero entry in a Jacobian matrix can be either a constant or an expression. For efficiency,
constant numbers and lambdified callables are stored separately. Constant numbers, therefore, can be loaded into
the sparse matrix pattern when a particular system is given.

.. warning::

    Data structure for the Jacobian storage has changed. Pending documentation update. Please check
    :py:mod:`andes.core.common.JacTriplet` class for more details.

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
instead of ``NumReduce``, ``NumRepeat`` and external variables.

..
    Atoms
    ANDES defines several types of atoms for building DAE models, including parameters, DAE variables,
    and service variables. Atoms can be used to build models and libraries, combined with discrete
    components and blocks.

