System
======

Overview
--------
System is the top-level class for organizing power system models and
orchestrating calculations. The full API reference of System is found at
:py:mod:`andes.system.System`.

Dynamic Imports
```````````````
System dynamically imports groups, models, and routines at creation. To add new
models, groups or routines, edit the corresponding file by adding entries
following examples.

.. autofunction:: andes.system.System.import_models
    :noindex:

.. autofunction:: andes.system.System.import_groups
    :noindex:

.. autofunction:: andes.system.System.import_routines
    :noindex:

Code Generation
```````````````
Under the hood, all models whose equations are provided in strings need be
processed to generate executable functions for simulations. We call this process
"code generation".  Code generation utilizes SymPy, a symbolic toolbox, and can
take up to one minute.

Code generation is automatically triggered upon the first ANDES run or whenever
model changes are detected. Code generation only needs to run once unless the
generated code is removed or model edits are detected. The generated code is
then stored and reused for speed up.

The generated Python code is called ``pycode``. It is a Python package (folder)
with each module (a `.py` file) storing the executable Python code and metadata
for numerical simulation. The default path to store ``pycode`` is
``HOME_DIR/.andes``, where ``HOME_DIR`` is one's `home directory`_.

.. _`home directory`: https://en.wikipedia.org/wiki/Home_directory

.. note::
    Code generation has been done if one has executed ``andes``,
    ``andes selftest``, or ``andes prepare``.

.. warning::
    For developers: when models are modified (such as adding new models or
    changing equation strings), code generation needs to be executed again
    for consistency. ANDES can automatically detect changes, and it can be
    manually triggered from command line using ``andes prepare -i``.

.. autofunction:: andes.system.System.prepare
    :noindex:

.. autofunction:: andes.system.System.undill
    :noindex:

DAE Storage
-----------

``System.dae`` is an instance of the numerical DAE class.

.. autofunction:: andes.variables.dae.DAE
    :noindex:

Model and DAE Values
--------------------
ANDES uses a decentralized architecture between models and DAE value arrays. In
this architecture, variables are initialized and equations are evaluated inside
each model. Then, ``System`` provides methods for collecting initial values and
equation values into ``DAE``, as well as copying solved values to each model.

The collection of values from models needs to follow protocols to avoid
conflicts. Details are given in the subsection Variables.

.. autofunction:: andes.system.System.vars_to_dae
    :noindex:

.. autofunction:: andes.system.System.vars_to_models
    :noindex:

.. autofunction:: andes.system.System._e_to_dae
    :noindex:

Matrix Sparsity Patterns
````````````````````````
The largest overhead in building and solving nonlinear equations is the building
of Jacobian matrices. This is especially relevant when we use the implicit
integration approach which algebraized the differential equations. Given the
unique data structure of power system models, the sparse matrices for Jacobians
are built **incrementally**, model after model.

There are two common approaches to incrementally build a sparse matrix. The
first one is to use simple in-place add on sparse matrices, such as doing

.. code:: python

    self.fx += spmatrix(v, i, j, (n, n), 'd')

Although the implementation is simple, it involves creating and discarding
temporary objects on the right hand side and, even worse, changing the sparse
pattern of ``self.fx``.

The second approach is to store the rows, columns and values in an array-like
object and construct the Jacobians at the end. This approach is very efficient
but has one caveat: it does not allow accessing the sparse matrix while
building.

ANDES uses a pre-allocation approach to avoid the change of sparse patterns by
filling values into a known the sparse matrix pattern matrix. System collects
the indices of rows and columns for each Jacobian matrix. Before in-place
additions, ANDES builds a temporary zero-filled `spmatrix`, to which the actual
Jacobian values are written later. Since these in-place add operations are only
modifying existing values, it does not change the pattern and thus avoids memory
copying. In addition, updating sparse matrices can be done with the exact same
code as the first approach.

Still, this approach creates and discards temporary objects. It is however
feasible to write a C function which takes three array-likes and modify the
sparse matrices in place. This is feature to be developed, and our prototype
shows a promising acceleration up to 50%.

.. autofunction:: andes.system.System.store_sparse_pattern
    :noindex:

Calling Model Methods
---------------------

System is an orchestrator for calling shared methods of models. These API
methods are defined for initialization, equation update, Jacobian update, and
discrete flags update.

The following methods take an argument `models`, which should be an
`OrderedDict` of models with names as keys and instances as values.

.. autofunction:: andes.system.System.init
    :noindex:

.. autofunction:: andes.system.System.e_clear
    :noindex:

.. autofunction:: andes.system.System.l_update_var
    :noindex:

.. autofunction:: andes.system.System.f_update
    :noindex:

.. autofunction:: andes.system.System.l_update_eq
    :noindex:

.. autofunction:: andes.system.System.g_update
    :noindex:

.. autofunction:: andes.system.System.j_update
    :noindex:


Configuration
-------------
System, models and routines have a member attribute `config` for model-specific
or routine-specific configurations. System manages all configs, including saving
to a config file and loading back.

.. autofunction:: andes.system.System.save_config
    :noindex:

.. warning::

    It is important to note that configs from files is passed to *model
    constructors* during instantiation. If one needs to modify config for a run,
    it needs to be done before instantiating ``System``, or before running
    ``andes`` from command line. Directly modifying ``Model.config`` may not
    take effect or have side effect as for the current implementation.

