.. _sec-command:

Command line
============

Basics
------

ANDES is invoked from the command line using the command ``andes``. Running
``andes`` without any input is equal to  ``andes -h`` or ``andes --help``. It
prints out a preamble with version and environment information, followed by
and help commands ::

        _           _         | Version 1.6.0
       /_\  _ _  __| |___ ___ | Python 3.9.10 on Linux, 03/12/2022 10:30:44 AM
      / _ \| ' \/ _` / -_|_-< |
     /_/ \_\_||_\__,_\___/__/ | This program comes with ABSOLUTELY NO WARRANTY.

    usage: andes [-h] [-v {1,10,20,30,40}]
                {run,plot,doc,misc,prepare,prep,selftest,st,demo} ...

    positional arguments:
    {run,plot,doc,misc,prepare,prep,selftest,st,demo}
                            [run] run simulation routine; [plot] plot
                            results; [doc] quick documentation; [misc] misc.
                            functions; [prepare] prepare the numerical code;
                            [selftest] run self test;

    optional arguments:
    -h, --help            show this help message and exit
    -v {1,10,20,30,40}, --verbose {1,10,20,30,40}
                            Verbosity level in 10-DEBUG, 20-INFO, 30-WARNING,
                            or 40-ERROR.

.. note::

    If the ``andes`` command is not found, it could be due to

    (1) missed steps in your installation process
    (2) errors during installation
    (3) forgot to activated the environment with ANDES

``andes`` accepts an optional arugment to control verbosity level. It is done
through ``-v LEVEL`` or ``--verbose LEVEL``, where ``level`` is a number.
Logging level by default is 20 (INFO) and can be chosen from:

- 1 (DEBUG with code location info)
- 10 (DEBUG)
- 20 (INFO)
- 30 (WARNING)
- 40 (ERROR)
- 50 (CRITICAL)

To show debugging outputs, use ``andes -v 10``, followed by top-level commands.
To only show warnings and errors, use ``andes -v 30``.

The top-level commands are ``{run,plot,doc,misc,prepare,selftest}``. Each
command contains a group of subcommands, which can be looked up with ``-h``. For
example, use ``andes run -h`` to look up the subcommands for ``andes run``.
Frequently used commands are explained below.

.. note::

    Some subcommands have shorthand names:

    - ``andes st`` is equivalent to ``andes selftest``
    - ``andes prep`` is equivalent to ``andes prepare``

andes selftest
--------------
After the installation, please run ``andes selftest`` from the command line to
test ANDES functionality. It might take a minute to run the full self-test
suite. An example output looks like

.. code-block:: console

    test_docs (test_1st_system.TestCodegen) ... ok
    ...
    ... (outputs are truncated)
    ...
    test_pflow_mpc (test_pflow_matpower.TestMATPOWER) ... ok
    ----------------------------------------------------------------------
    Ran 60 tests in 10.109s

    OK

There may be more test than what is shown above. Make sure that all tests have
passed.

ANDES receives frequent updates. After each update, please run ``andes
st`` to confirm the functionality. The command also makes sure the
generated code is up to date. See `andes prepare`_ for more details on
automatic code generation.

.. note::

    There is a quick mode to test ANDES by skipping code generation. This should
    only be used when you are certain that there is no modification to models
    between the last code generation and now.

    The quick mode is invoked by ``andes st -q``.

.. _`andes prepare`:

andes prepare
-----------------

The symbolically defined models in ANDES need to be generated into numerical
code for simulation. The code generation process is *automatic* the first time you
use ANDES to run any case study. It takes 10 seconds to one minute to generate
the code depending on your platform. When done, no code generation is needed in
your future use untill you modify the models.

It is also possible to generate the code manually with ``andes prepare`` or
``andes prep``.  In addition, ``andes selftest`` automatically calls the
code generation.

.. note::

    Generated code files are stored in Python code in ``$HOME/.andes/pycode``.
    While being human-readable, they are not human-friendly and should only be
    consulted during low-level debugging.

The default code generation mode is known as the "quick mode". It skips the
generation of :math:`\LaTeX`-formatted equations, which are only useful in
documentation and the interactive mode.

Option ``-i`` or ``--incremental`` can be used to speed up code generation
during model development. ``andes prepare -i`` only generates code for
models that are detected with changes since the last code generation.

.. warning::

    To developers:

    ``andes prepare -i`` needs to be called following model changes, such as
    equation modification and adding variables. Otherwise, due to mismatches in
    model and code, simulation results will not reflect the new changes, at
    best, or even lead to unexpected errors

ANDES supports precompiling the generated Python code using Numba. See
:ref:`numba-compilation`.

andes run
-------------
``andes run`` is the entry point for power system analysis routines. The
full list of options can be printed with ``andes run -h``. ``andes run``
takes one positional argument, ``filename``, along with other optional
keyword arguments. ``filename`` is the path to cases, either relative or
absolute.

- **Relative path**: ``andes run kundur_full.xlsx``, e.g., uses a relative path.
  It works only if ``kundur_full.xlsx`` exists in the *working directory* of
  the command line.

- **Absolute path**: ``andes run /Users/hcui7/kundur_full.xlsx`` (on macOS) or
  ``andes run C:/Users/hcui7/kundur_full.xlsx`` (on Windows) use absolute
  paths to the case files. They do not depend on the command-line current
  directory.

.. note ::

    When working with the command line, use ``cd`` to change directory to the
    folder containing your test case. Spaces in folder and file names need to be
    escaped properly, so it's generally advised to *avoid spaces in file and
    folder names*.

To find out your current working directory, look for the line below the ANDES
preamble that reads like

::

    Working directory: "/home/hacui/repos/andes/andes/cases/kundur"

Input path
..........
ANDES allows one to specify the path to look for the case file instead of the
working directory. This is done by using the ``-p`` or ``--input-path`` option.
For example, if ``kundur_full.xlsx`` is in folder ``/home/hacui/cases``, one can
do

.. code-block:: console

    andes run kundur_full.xlsx -p /home/hacui/cases

The argument passed to ``-p`` or ``--input-path`` can also be a relative path.
If you need further help understanding paths, please consult other online
articles.

Multiprocessing
...............

ANDES takes multiple files inputs or wildcard. Multiprocessing will be
triggered if more than one valid input files are passed to ``filename``.


- Multiple files: to run the power flow for ``kundur_full.xlsx`` and
  ``kundur_motor.xlsx`` simultaneously, one can do

.. code-block:: console

    andes run kundur_full.xlsx kundur_motor.xlsx

The output will look like ::

    Working directory: "/home/hacui/repos/andes/andes/cases/kundur"
    -> Processing 2 jobs on 12 CPUs.
    Process 0 for "kundur_full.xlsx" started.
    Process 1 for "kundur_motor.xlsx" started.
    Log saved to "/tmp/andes/andes-uopdutii/andes.log".
    -> Multiprocessing finished in 2.4680 seconds.

- Wildcard: to run power flow for files with a prefix of ``kundur_`` and a suffix
  (file extension) of ``.xlsx``, run

.. code-block:: console

    andes run kundur_*.xlsx

Case files with such name pattern, including ``kundur_full.xlsx`` and
``kundur_motor.xlsx``, among others, will be processed.

Option ``--ncpu NCPU`` can be used to specify the maximum number of parallel
processes. By default, all cores will be used. A small number can be
specified to increase operating system responsiveness.

Routine
.......
Option ``-r`` or ``-routine`` is used for specifying the analysis routine,
followed by the routine name. Available routine names include

- ``pflow`` for power flow calculation
- ``tds`` for time domain simulation
- ``eig`` for eigenvalue analysis

If ``-r`` is not given, the power flow calculation routine will be called.
There are routine specific options that can be passed to ``andes run`` and are
discussed next.

Each routine has a list of configuration options (called "config") to
control their behaviors. Config needs to be distinguished from command-line
options as not all config options are available in the command-line.
Refer to :ref:`configuration` for details.

Power flow
..........

.. note::

    Examples in the following will utilize the ``kundur_full.xlsx`` test case.
    If you have cloned the ANDES repository, it can be found in
    ``andes/cases/kundur`` in  the source code folder. You can also download it
    from
    `here <https://github.com/cuihantao/andes/raw/master/andes/cases/kundur/kundur_full.xlsx>`_.

To run power flow, change to the directory containing ``kundur_full.xlsx``, and
execute the following in the command line:

.. code:: bash

    andes run kundur_full.xlsx

Alternatively, the full path to the case file is also recognizable, such as

.. code:: bash

    andes run /home/user/andes/cases/kundur/kundur_full.xlsx

The power flow report will be saved to the current directory where ANDES is
invoked. The report contains four sections:

1) system statistics,
2) ac bus and dc node data
3) ac line data, and
4) results of other algebraic variables and state variables.

By default, the power flow routine is configured to use full Newton Raphson
method, and reactive power limits are not checked. To change these config, edit
the config file by referring to ``andes doc PFlow`` and ``andes doc PV``.

Following power flow, ANDES does not initialize dynamic models to save time.
When developing dynamic models, one can enable the initialization by setting in
the config file ::

    [PFlow]
    init_tds = 1

Time-domain simulation
......................

To run the time domain simulation (TDS) for ``kundur_full.xlsx``, run

.. code:: bash

    andes run kundur_full.xlsx -r tds

The output looks like::

    Parsing input file "kundur_full.xlsx"...
    Input file parsed in 0.1533 seconds.
    System internal structure set up in 0.0174 seconds.
    -> System connectivity check results:
    No islanded bus detected.
    System is interconnected.
    Each island has a slack bus correctly defined and enabled.

    -> Power flow calculation
    Sparse solver: KLU
    Solution method: NR method
    Numba compilation initiated with caching.
    Power flow initialized in 0.1428 seconds.
    0: |F(x)| = 14.9282832
    1: |F(x)| = 3.608627841
    2: |F(x)| = 0.1701107882
    3: |F(x)| = 0.002038626956
    4: |F(x)| = 3.745103977e-07
    Converged in 5 iterations in 0.0014 seconds.
    Report saved to "kundur_full_out.txt" in 0.0004 seconds.

    -> Time Domain Simulation Summary:
    Sparse Solver: KLU
    Simulation time: 0.0-20.0 s.
    Fixed step size: h=33.33 ms. Shrink if not converged.
    Numba compilation initiated with caching.
    Initialization for dynamics completed in 0.0626 seconds.
    Initialization was successful.
    <Toggler 1>: Line.Line_8 status changed to 0 at t=2.0 sec.
    100%|########################################| 100/100 [00:00<00:00, 241.53%/s]
    Simulation completed in 0.4141 seconds.
    Outputs to "kundur_full_out.lst" and "kundur_full_out.npz".
    Outputs written in 0.0171 seconds.
    -> Single process finished in 0.8890 seconds.

The output contains the key information for the simulation, such as solver
name and step size. It prints out the disturbance information, the trip of
Line ``Line_8`` at time ``t=2.0 sec``.

There are a few places needing to be noted:

1. Make sure the power flow calculation is successful. Otherwise, there is no
   good starting point for dynamic simulation.
2. Make sure no suspect initialization error is found. Otherwise, the system
   will not be at steady state even before disturbances.

TDS writes two output files: a variable list file ``kundur_full_out.lst``,
and a compressed NumPy data file ``kundur_full_out.npz``:

- List file: it is a plain-text file with three columns: variable indices,
  variable name in plain text, and variable name in the :math:`\LaTeX`
  format. The variable indices are needed to plot the needed variable.

- Data file: it is a zipped NumPy binary file. Although not directly editable,
  it can be used for plotting or can be converted to a CSV file. See the
  subsection on `andes plot`_.

There are TDS-specific options that can be passed to ``andes run``:

- ``--tf TF``: the final time of the simulation. ``TF`` should be a number in
  seconds. By default, it is set to 20.0.
- ``--addfile ADDFILE``: specify an additional data file. This is currently used
  to supply PSS/E dyr file in addition to a raw file.
- ``--flat``: turn on "flat run" mode to ignore all disturbances. The simulation
  will be performed up to the end time.
- ``--no-pbar``: turn off progress bar.
- ``--from-csv FROM_CSV``: use data from a CSV file to perform mock simulation.
  The CSV file should be in the format of ``andes plot --to-csv``.

Disable output
..............
Output to files can be disabled with ``--no-output`` or ``-n``. It is useful
when computation is needed but results can be discarded. It is also useful when
results are processed in memory, combined with the ``--shell`` option discussed
next.

IPython shell
.............
The ANDES CLI will exit to the system shell when finished running. It is
sometimes useful to script in Python to quickly process the simulation results
in memory, such as plotting. ANDES can exit to the IPython shell with
``--shell`` or ``-s``. For example:

.. code:: bash

    andes run kundur_full.xlsx -r tds -s -n

Note the ``-n`` is optional to disable file output. The terminal output will
look like ::

    <Toggler 1>: Line.Line_8 status changed to 0 at t=2.0 sec.
    100%|#########################################| 100/100 [00:00<00:00, 246.07%/s]
    Simulation completed in 0.4064 seconds.
    Outputs to "kundur_full_out.lst" and "kundur_full_out.npz".
    Outputs written in 0.0171 seconds.
    -> Single process finished in 0.8796 seconds.
    IPython: Access System object in variable `system`.
    Python 3.9.10 | packaged by conda-forge | (main, Feb  1 2022, 21:24:11)
    Type 'copyright', 'credits' or 'license' for more information
    IPython 8.1.1 -- An enhanced Interactive Python. Type '?' for help.

    In [1]:

A prompt will appear following ``In [1]:`` to indicate an IPython shell.
If the test case file is parsed without error, the system object will be stored
in variable ``system``, i.e.

::

    In [1]: system
    Out[1]: <andes.system.System at 0x7fc1cd992790>

Python commands can be executed thereafter. To exit, type ``exit`` and press
enter.


.. _format-converter:

Format converter
................

ANDES uses the Excel format to store power system data in the ANDES semantics.
In addition, multiple input formats are recognized and can be converted to the
ANDES ``xlsx`` format. Converting data into the ANDES has pros and cons:

- Pros:
  - The data can be readily edited with an Excel-like tool
  - Data for models unique to ANDES can be readily added to the ``xlsx`` file

- Cons:
  - Conversion from ANDES ``xlsx`` back to the original format is not supported

.. note::

    It is recommended to stay with the original data format to maximize
    compatibility when no ANDES-specific models are used.

Format conversion is done through ``--convert FORMAT`` or ``-c FORMAT``, where
``FORMAT`` is the output format. For now, the following formats are supported:

- ``xlsx``: an Excel spread sheet format with ANDES-specific data. It is
  not compatible with ``xlsx`` with datafrom other tools such as
  `Pandapower <https://www.pandapower.org>`_.
- ``json``: a JSON plain-text file with ANDES-specific data. Likewise, it is
  unlikely to be compatible with JSON from other power system tools. JSON is
  much faster to parse than ``xlsx`` but not as friendly to edit.

To convert ``kundur_full.xlsx``, for example, to the ``json`` format, run

.. code:: bash

    andes run kundur_full.xlsx --convert json

The output messages will look like ::

    Parsing input file "kundur_full.xlsx"...
    Input file parsed in 0.1576 seconds.
    System internal structure set up in 0.0175 seconds.
    JSON file written to "kundur_full.json"
    Format conversion completed in 0.0053 seconds.
    -> Single process finished in 0.2582 seconds.

Note that ``--convert`` will only create sheets for existing models.

The converter works with any input formats that are currently supported. These
include:

- ``.m``: MATPOWER case file
- ``.raw`` and ``.dyr``: PSS/E raw and dyr files
- ``.xlsx``: Excel spreadsheet file with ANDES data
- ``.json``: JSON plain-text file with ANDES data

PSS/E inputs
............
To work with PSS/E input files (.raw and .dyr), one need to provide the raw file
through ``casefile`` and pass the dyr file through ``--addfile``.
For example, in ``andes/cases/kundur``, one can run the power flow using

.. code:: bash

    andes run kundur.raw

and run a no-disturbance time-domain simulation using

.. code:: bash

    andes run kundur.raw --addfile kundur_full.dyr -r tds

.. note::
    If one wants to modify the parameters of models that are supported
    by both PSS/E and ANDES, one can directly
    edit those dynamic parameters in the ``.raw`` and ``.dyr`` files
    to maintain interoperability with other tools.

To create add a disturbance, there are two options. The recommended option is to
convert the PSS/E data into an ANDES xlsx file, edit it and run (see the
previous subsection). The alternative approach is documented in
:ref:`creating disturbances`.

Profiling
.........
Profiling is useful for analyzing the computation time and code efficiency.
Option ``--profile`` enables the profiling of ANDES execution. The profiling
output will be written in two files in the current folder, one ending with
``_prof.txt`` and the other one with ``_prof.prof``.

The text file can be opened with a text editor, and the ``.prof`` file can be
visualized with ``snakeviz``, which can be installed with ``pip install
snakeviz``.

If the output is disabled, profiling results will be printed to stdio.

andes plot
----------
.. _`andes plot`:

``andes plot`` is the command-line tool for plotting. It currently supports
time-domain simulation data. Three positional arguments are required, and a
dozen of optional arguments are supported.

positional arguments:

    +----------------+----------------------------------------------------------------------+
    | Argument       |             Description                                              |
    +================+======================================================================+
    | filename       |    simulation output file name, which should end with                |
    |                |    `out`. File extension can be omitted.                             |
    +----------------+----------------------------------------------------------------------+
    | x              |    the X-axis variable index, typically 0 for Time                   |
    +----------------+----------------------------------------------------------------------+
    | y              |    Y-axis variable indices. Space-separated indices or a             |
    |                |    colon-separated range is accepted                                 |
    +----------------+----------------------------------------------------------------------+

For the list of optional arguments, see the output of ``andes plot -h``.

To plot the generator speed variable ``omega`` of GENROU_1 ``omega GENROU 1``
versus time, one way is to supply the variable indices found in the ``.lst``
output file. The index of the variable ``omega GENROU 1`` is found to be ``5``,
and Time is found to be ``0``, so the plot command should be

.. code:: bash

    andes plot kundur_full_out.lst 0 5

where ``kundur_full_out.lst`` is list file name, ``0`` is the index of ``Time``
for the x-axis, and ``5`` is the index of ``omega GENROU 1``. Note that for the
the file name, either ``kundur_full_out.lst`` or ``kundur_full_out.npz`` works
as the program will automatically extract the file name.

The y-axis variable indices can also be specified as a Python range. For
example, ``andes plot kundur_full_out.npz 0 2:21:6`` will plot the variables
with indices 2, 8, 14 and 20.

It can become tedious to look up the indices of variables in the ``.lst`` file.
``andes plot`` supports ``--xargs`` or ``-a`` for searching for variable indices
and passing them as arguments to ``andes plot``. See Examples - "Using CLI from
Notebook".

LaTeX rendering
...............

``andes plot`` will attempt to render with :math:`\LaTeX` if ``dvipng`` program
is in the search path. Figures rendered by :math:`\LaTeX` has
publication-quality aesthetics for symbols but takes considerably longer time.
In case :math:`\LaTeX` is available but fails (frequently happens on Windows),
the option ``-d`` can be used to disable :math:`\LaTeX` rendering.

.. _andes_doc:

andes doc
---------
``andes doc`` is a handy tool to look up model, routine and config
documentation. Model documentation include the descriptions of parameters,
variables, and configs. A pretty-print version is available online in
:ref:`modelref`.

The basic usage of ``andes doc`` is to provide a model name or a routine name as
the positional argument. For a model, it will print out model parameters,
variables, and equations to the stdio. For a routine, it will print out fields
in the Config file.

.. note::

    For full model documentation, visit :ref:`modelref`.

For example, to check the parameters for model ``Toggler``, run

.. code-block:: shell-session

    $ andes doc Toggler
    Model <Toggler> in Group <TimedEvent>

        Time-based connectivity status toggler.

    Parameters

     Name  |         Description          | Default | Unit |    Type    | Properties
    -------+------------------------------+---------+------+------------+-----------
     u     | connection status            | 1       | bool | NumParam   |
     name  | device name                  |         |      | DataParam  |
     model | Model or Group of the device |         |      | DataParam  | mandatory
           | to control                   |         |      |            |
     dev   | idx of the device to control |         |      | IdxParam   | mandatory
     t     | switch time for connection   | -1      |      | TimerParam | mandatory
           | status                       |         |      |            |

To list all supported models, run

.. code-block:: shell-session

    $ andes doc -l
    Supported Groups and Models

         Group       |                   Models
    -----------------+-------------------------------------------
     ACLine          | Line
     ACTopology      | Bus
     Collection      | Area
     DCLink          | Ground, R, L, C, RCp, RCs, RLs, RLCs, RLCp
     DCTopology      | Node
     Exciter         | EXDC2
     Experimental    | PI2
     FreqMeasurement | BusFreq, BusROCOF
     StaticACDC      | VSCShunt
     StaticGen       | PV, Slack
     StaticLoad      | PQ
     StaticShunt     | Shunt
     SynGen          | GENCLS, GENROU
     TimedEvent      | Toggler, Fault
     TurbineGov      | TG2, TGOV1

To view the Config fields for a routine, run

.. code-block:: shell-session

    $ andes doc TDS
    Config Fields in [TDS]

      Option   | Value |                  Info                  | Acceptable values
    -----------+-------+----------------------------------------+-------------------
     sparselib | klu   | linear sparse solver name              | ('klu', 'umfpack')
     tol       | 0.000 | convergence tolerance                  | float
     t0        | 0     | simulation starting time               | >=0
     tf        | 20    | simulation ending time                 | >t0
     fixt      | 0     | use fixed step size (1) or variable    | (0, 1)
               |       | (0)                                    |
     shrinkt   | 1     | shrink step size for fixed method if   | (0, 1)
               |       | not converged                          |
     tstep     | 0.010 | the initial step step size             | float
     max_iter  | 15    | maximum number of iterations           | >=10


.. _andes-misc:

andes misc
----------
``andes misc`` contains miscellaneous functions, such as configuration and
output cleaning.

Configuration
.............
ANDES uses a configuration file to set runtime configs for the system routines,
and models. ``andes misc --save-config`` saves all configs to a file. By
default, it saves to ``$HOME/.andes/andes.conf`` file, where ``$HOME`` is the
path to your home directory.

With ``andes misc --edit-config``, you can edit ANDES configuration handy. The
command will automatically save the configuration to the default location if not
exist. The shorter version ``--edit`` can be used instead as Python matches it
with ``--edit-config``.

You can pass an editor name to ``--edit``, such as ``--edit vim``. If the editor
name is not provided, it will use the following defaults: - Microsoft Windows:
notepad. - GNU/Linux: the ``$EDITOR`` environment variable, or ``vim`` if not
exist.

For macOS users, the default is vim. If not familiar with vim, you can use nano
with ``--edit nano`` or TextEdit with ``--edit "open -a TextEdit"``.

Cleanup
.......
``andes misc -C, --clean``

Option to remove any generated files. Removes files with any of the following
suffix: ``_out.txt`` (power flow report), ``_out.npy`` (time domain data),
``_out.lst`` (time domain variable list), and ``_eig.txt`` (eigenvalue report).

Version
.......
Check the version of ANDES and the core packages it uses, use

.. code:: bash

    andes misc --version

Please include the output in your bug report.