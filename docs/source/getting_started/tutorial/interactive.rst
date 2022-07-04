
Scripting
=========
This section is a tutorial for using ANDES in an interactive/scripting
environment. All scripting shells are supported, including Python shell,
IPython, Jupyter Notebook and Jupyter Lab. The examples below uses Jupyter
Notebook.

Jupyter Notebook
----------------
Jupyter notebook is a convenient tool to run Python code and present results.
Jupyter notebook can be installed with

.. code:: bash

    conda install jupyter notebook

After the installation, change directory to the folder where you wish to store
notebooks, then start the notebook with

.. code:: bash

    jupyter notebook

A browser window should open automatically with the notebook browser loaded. To
create a new notebook, use the "New" button near the upper-right corner.

.. note::

    In the following, code that starts with ``>>>`` are Python code. and should
    be run inside Python, IPython, or Jupyter Notebook. Python code should not
    be entered into Anaconda Prompt or Linux shell.

Import
------
Like other Python libraries, ANDES needs to be imported into an interactive
scripting Python environment.

.. code:: python

    >>> import andes
    >>> andes.config_logger()

Verbosity
---------
If you are debugging ANDES, you can enable debug messages with

.. code:: python

    >>> andes.config_logger(stream_level=10)

or simply

.. code:: python

    >>> andes.config_logger(10)

The ``stream_level`` uses the same verbosity levels as for the command-line. If
not explicitly enabled, the default level 20 (INFO) will apply.

To set a new logging level for the current session, call ``config_logger`` with
the desired new levels.

Making a System
---------------
Before running studies, an :py:mod:`andes.system.System` object needs to be
create to hold the system data. The System object can be created by passing the
path to the case file the entry-point function.

There are multiple ways to create such object, and :py:mod:`andes.main.run` is
the most convenient way. For example, to run the file ``kundur_full.xlsx`` in
the same directory as the notebook, use

.. code:: python

    >>> ss = andes.run('kundur_full.xlsx')

This function will parse the input file, run the power flow, and return the
system as an object. Outputs will look like ::

    Parsing input file </Users/user/notebooks/kundur/kundur_full.xlsx>
    Input file kundur_full.xlsx parsed in 0.4172 second.
    -> Power flow calculation with Newton Raphson method:
    0: |F(x)| = 14.9283
    1: |F(x)| = 3.60859
    2: |F(x)| = 0.170093
    3: |F(x)| = 0.00203827
    4: |F(x)| = 3.76414e-07
    Converged in 5 iterations in 0.0222 second.
    Report saved to </Users/user/notebooks/kundur_full_out.txt> in 0.0015 second.
    -> Single process finished in 0.4677 second.

In this example, ``ss`` is an instance of ``andes.System``. It contains member
attributes for models, routines, and numerical DAE.

Naming convention for the ``System`` attributes are as follows

- Model attributes share the same name as class names. For example, ``ss.Bus``
  is the ``Bus`` instance, and ``ss.GENROU`` is the ``GENROU`` instance.
- Routine attributes share the same name as class names. For example,
  ``ss.PFlow`` and ``ss.TDS`` are the routine instances.
- The numerical DAE instance is in lower case ``ss.dae``.

To work with PSS/E inputs, refer to :ref:`scripting_examples` - "Working with
Data".

.. note::
    :py:mod:`andes.main.run` can accept multiple input files for multiprocessing.
    They can be passed as a list of strings to the first positional argument.

Passing options
...............
``andes.run()`` can accept options that are available to the command-line
``andes run``. Options need to be passed as keyword arguments to ``andes.run()``
in addition to the positional argument for the test case. For example, setting
``no_output`` to ``True`` will disable all file outputs. When scripting, one can
do

.. code:: python

    >>> ss = andes.run('kundur_full.xlsx', no_output=True)

which is equivalent to the following shell command:

.. code:: bash

    andes run kundur_full.xlsx --no-output

Please note that the dash between ``no`` and ``output`` needs to be replaced
with an underscore for scripting. This is the convention in Python's argument
parser.

Another example is to specify a folder for output files. By default, outputs
will be saved to the folder where Python is run (or where the notebook is run).
In case you need to organize outputs, a path prefix can be passed to
``andes.run()`` through ``output_path``:

.. code:: python

    >>> ss = andes.run('kundur_full.xlsx', output_path='outputs/')

which will put outputs into folder ``outputs`` relative to the current path. You
can also supply an absolute path to ``output_path``.

The next example is to specify the simulation time for a time-domain simulation.
There are multiple ways to implement it (see :ref:`scripting_examples`), and one
way is to pass the end time (in sec) through argument ``tf`` and set the
``routine`` to ``tds``:

.. code:: python

    >>> ss = andes.run('kundur_full.xlsx', routine='tds', tf=5)

which will set the simulation time to 5 seconds.

.. note::

    While ``andes run`` accepts single-letter alias for the option, such as
    ``andes run -n`` for ``andes run --no-output``, ``andes.run()`` can only
    work with the full option name (with hyphen replaced by underscore)

Load Only
.........
In many workflows, one will simulate many scenarios with largely identical
system data. A base case can be loaded and modified to create scenarios in
memory. See Example "Working with Data" for details

Inspecting Parameter
--------------------

DataFrame
.........
Parameters for the loaded system can be readily inspected in Jupyter Notebook
using Pandas.

Parameters for a model instance can be retrieved in a DataFrame using the
``as_df()`` method on the model instance. For example, to view the parameters of
``Bus``, use

.. code:: python

    >>> ss.Bus.as_df()

A table will be printed with the columns being parameters and the rows being Bus
devices/instances. For a system that has been setup, parameters have been
converted to per unit values in the system base specified by ``ss.config.mva``.
The per-unit values in the system base will be used in computation as all
computation in ANDES uses system-base per-unit data.

To view the original input values, use the ``as_df(vin=True)`` method. For
example, to view the system-base per unit value of ``GENROU``, use

.. code:: python

    >>> ss.GENROU.as_df(vin=True)

Parameter in the table is the same as that in the input file without any
conversion. Some input data, by convention, are given as per unit in the device
base; see :ref:`per_unit_system` for details.

Note that :py:meth:`andes.core.modeldata.ModelData.as_df` returns a *view*.
Modifying the returned dataframe *will not* affect the original data used for
simulation. To modify the data, see Example "Working with Data".

Running Studies
---------------

Three routines are currently supported: PFlow, TDS and EIG. Each routine
provides a ``run()`` method to execute. The System instance contains member
attributes having the same names. For example, to run the time-domain simulation
for ``ss``, use

.. code:: python

    >>> ss.TDS.run()

To change configuration for routines, one can set the attribute before
calling run. For example, to change the end time to 5 sec, one can do

.. code:: python

    >>> ss.TDS.config.tf = 5
    >>> ss.TDS.run()

Note that not all config changes are respected. Some config values
are used while creating the routine instance. For config changes
that does not necessarily have to be done on-the-fly, it is recommended to
edit the config file.

Checking Exit Code
------------------
``andes.System`` contains field ``exit_code`` for checking if error occurred in
run time. A normal completion without error should always have ``exit_code ==
0``. One should read output messages carefully and check the exit code, which is
particularly useful for batch simulations.

Error may occur in any phase - data parsing, power flow, or simulation. To
diagnose, split the simulation steps and check the outputs from each one.

Plotting TDS Results
--------------------
TDS comes with a plotting utility for scripting usage. After running the
simulation, a ``plotter`` attributed will be created for ``TDS``. To use the
plotter, provide the attribute instance of the variable to plot. For example, to
plot all the generator speed, use

.. code:: python

    >>> ss.TDS.plotter.plot(ss.GENROU.omega)

.. note::

    If you see the error

        AttributeError: 'NoneType' object has no attribute 'plot'

    You will need to manually load plotter with

    .. code:: python

        >>> ss.TDS.load_plotter()

Optional indices is accepted to choose the specific elements to plot. It can be
passed as a tuple through the ``a`` argument

.. code:: python

    >>> ss.TDS.plotter.plot(ss.GENROU.omega, a=(0, ))

In the above example, the speed of the "zero-th" generator will be plotted.

Scaling
.......
A lambda function can be passed to argument ``ycalc`` to scale the values. This
is useful to convert a per-unit variable to nominal. For example, to plot
generator speed in Hertz, use

.. code:: python

    >>> ss.TDS.plotter.plot(ss.GENROU.omega, a=(0, ),
                            ycalc=lambda x: 60*x,
                            )

Formatting
..........
A few formatting arguments are supported:

- ``grid = True`` to turn on grid display
- ``greyscale = True`` to switch to greyscale
- ``ylabel`` takes a string for the y-axis label

Extracting Data
---------------
One can extract data from ANDES for custom plotting. Variable names can be
extracted from the following fields of ``ss.dae``:

Un-formatted names (non-LaTeX):

- ``x_name``: state variable names
- ``y_name``: algebraic variable names
- ``xy_name``: state variable names followed by algebraic ones

LaTeX-formatted names:

- ``x_tex_name``: state variable names
- ``y_tex_name``: algebraic variable names
- ``xy_tex_name``: state variable names followed by algebraic ones

These lists only contain the variable names used in the current analysis
routine. If you only ran power flow, ``ss.dae.y_name`` will only contain the
power flow algebraic variables, and ``ss.dae.x_name`` will likely be empty.
After initializing time-domain simulation, these lists will be extended to
include all variables used by TDS.

In case you want to extract the discontinuous flags from TDS, you can set
``store_z`` to ``1`` in the config file under section ``[TDS]``. When enabled,
discontinuous flag names will be populated at

- ``ss.dae.z_name``: discontinuous flag names
- ``ss.dae.z_tex_name``: LaTeX-formatted discontinuous flag names

If not enabled, both lists will be empty.

Power flow solutions
....................
The full power flow solutions are stored at ``ss.dae.xy`` after running power
flow (and before initializing dynamic models). You can extract values from
``ss.dae.xy``, which corresponds to the names in ``ss.dae.xy_name`` or
``ss.dae.xy_tex_name``.

If you want to extract variables from a particular model, for example, bus
voltages, you can directly access the ``v`` field of that variable

.. code:: python

    >>> import numpy as np
    >>> voltages = np.array(ss.Bus.v.v)

which stores a **copy** of the bus voltage values. Note that the first ``v`` is
the voltage variable of ``Bus``, and the second ``v`` stands for *value*. It is
important to make a copy by using ``np.array()`` to avoid accidental changes to
the solutions.

If you want to extract bus voltage phase angles, do

.. code:: python

    >>> angle = np.array(ss.Bus.a.v)

where ``a`` is the field name for voltage angle.

To find out names of variables in a model, use command ``andes doc`` or refer to
:ref:`modelref`.

Time-domain data
................

Time-domain simulation data will be ready when simulation completes. It is
stored in ``ss.dae.ts``, which has the following fields:

- ``txyz``: a two-dimensional array. The first column is time stamps, and the
  following are variables. Each row contains all variables for that time step.
- ``t``: all time stamps.
- ``x``: all state variables (one column per variable).
- ``y``: all algebraic variables (one column per variable).
- ``z``: all discontinuous flags (if enabled, one column per flag).

If you want the output in pandas DataFrame, call

.. code:: python

    ss.dae.ts.unpack(df=True)

Dataframes are stored in the following fields of ``ss.dae.ts``:

- ``df``: dataframe for states and algebraic variables
- ``df_z``: dataframe for discontinuous flags (if enabled)

For both dataframes, time is the index column, and each column correspond to one
variable.

.. note::

    Looking to extract data for a single variable? See :ref:`scripting_examples`
    - "Working with Data".

Pretty Print of Equations
----------------------------------------
Each ANDES models offers pretty print of :math:`\LaTeX`-formatted equations in
the jupyter notebook environment.

To use this feature, symbolic equations need to be generated in the current
session using

.. code:: python

    import andes ss = andes.System() ss.prepare()

Or, more concisely, one can do

.. code:: python

    import andes ss = andes.prepare()

This process may take a few minutes to complete. To save time, you can
selectively generate it only for interested models. For example, to generate for
the classical generator model ``GENCLS``, do

.. code:: python

    import andes ss = andes.System() ss.GENROU.prepare()

Once done, equations can be viewed by accessing
``ss.<ModelName>.syms.<PrintName>``, where ``<ModelName>`` is the model name,
and ``<PrintName>`` is the equation or Jacobian name.

.. Note ::

    Pretty print only works for the particular ``System`` instance whose
    ``prepare()`` method is called. In the above example, pretty print only
    works for ``ss`` after calling ``prepare()``.

Supported equation names include the following:

- ``xy``: variables in the order of `State`, `ExtState`, `Algeb` and `ExtAlgeb`
- ``f``: the **right-hand side of** differential equations :math:`\mathbf{M}
  \dot{\mathbf{x}} = \mathbf{f}`
- ``g``: implicit algebraic equations :math:`0 = \mathbf{g}`
- ``df``: derivatives of ``f`` over all variables ``xy``
- ``dg``: derivatives of ``g`` over all variables ``xy``
- ``s``: the value equations for `ConstService`

For example, to print the algebraic equations of model ``GENCLS``, one can use
``ss.GENCLS.syms.g``.

Finding Help
------------

docstring
.........

To find out how a Python class, method, or function should be used, use the
built-in ``help()`` function. This will print out the docstring of the
class/method/function. For example, to check how the ``get`` method of
``GENROU`` should be called, do

.. code:: python

    help(ss.GENROU.get)

In Jupyter notebook, this can be simplified into ``?ss.GENROU.get`` or
``ss.GENROU.get?``.

Please report issues if you find missing docstring.

Model docs
..........

Model docs can be shown by printing the return of ``doc()``. For example, to
check the docs of ``GENCLS``, do

.. code:: python

    print(ss.GENCLS.doc())

It is the same as calling ``andes doc GENCLS`` from the command line.
Likewise, a pretty-print version is available online in :ref:`modelref`.

.. _formats:
