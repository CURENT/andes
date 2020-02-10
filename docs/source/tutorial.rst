.. _tutorial:

********
Tutorial
********
This chapter describes the mose common usages.
ANDES can be used as a command-line tool or as a library.
The command-line interface (CLI) comes handy to run studies.
As a library, it can be used interactively in the IPython shell or the Jupyter Notebook.

Please use the following shortcuts if you are looking for particular topics.

.. _sec-command:

Command Line Usage
=======================

Basic Usage
-----------

ANDES is invoked from the command line using the command ``andes``.
Running ``andes`` without any input is equal to  ``andes -h`` or ``andes --help``.
It prints out a preamble with version and environment information and help commands::

    ANDES 0.6.8 (Git commit id 0ace2bc0, Python 3.7.6 on Darwin)
    Session: hcui7, 02/09/2020 08:34:35 PM

    usage: andes [-h] [-v {10,20,30,40,50}] {run,plot,misc,prepare,selftest} ...

    positional arguments:
      {run,plot,misc,prepare,selftest}
                            [run]: run simulation routine; [plot]: plot simulation
                            results; [prepare]: run the symbolic-to-numeric
                            preparation; [misc]: miscellaneous functions.

    optional arguments:
      -h, --help            show this help message and exit
      -v {10,20,30,40,50}, --verbose {10,20,30,40,50}
                            Program logging level. Available levels are 10-DEBUG,
                            20-INFO, 30-WARNING, 40-ERROR or 50-CRITICAL. The
                            default level is 20-INFO.


The first level of commands are chosen from ``{run,plot,misc,prepare,selftest}``. Each command contains a group
of subcommands, which can be looked up with ``-h``. For example, use ``andes run -h`` to look up the subcommands
in ``run``. The most commonly used commands will be explained in the following.

andes selftest
--------------
After installing ANDES, it is encouraged to use ``andes selftest`` to run tests and check the basic functionality.
It might take a minute to run the whole self-test suite. Results are printed as the tests proceed. An example
output looks like ::

    ANDES 0.6.8 (Git commit id 0ace2bc0, Python 3.7.6 on Darwin)
    Session: hcui7, 02/09/2020 08:44:07 PM

    test_docs (test_1st_system.TestSystem) ... ok
    test_limiter (test_discrete.TestDiscrete) ... ok
    test_sorted_limiter (test_discrete.TestDiscrete) ... ok
    test_switcher (test_discrete.TestDiscrete) ... ok
    test_pflow_mpc (test_pflow_matpower.TestMATPOWER) ... ok
    test_count (test_pjm.Test5Bus) ... ok
    test_idx (test_pjm.Test5Bus) ... ok
    test_names (test_pjm.Test5Bus) ... ok
    test_pflow (test_pjm.Test5Bus) ... ok
    100%|██████████████████████████████████████████| 100/100 [00:02<00:00, 39.29%/s]
    ok

    ----------------------------------------------------------------------
    Ran 10 tests in 21.289s

    OK

Test cases can grow, and there could be more cases than above. Make sure that all tests have passed.

andes prepare
-----------------
The symbolically defined models in ANDES need to be generated into numerical code for simulation.
The code generation can be manually called with ``andes prepare``.
Generated code are stored in folder ``.andes/calls.pkl`` in your home directory.
In addition, ``andes selftest`` implicitly calls the code generation.
If you are using ANDES as a package in the user mode, you won't need to call it again.

For developers, ``andes prepare`` needs to be called immediately following any model equation
modification. Otherwise, simulation results will not reflect the new equations and will likely lead to an error.

Option ``-q`` or ``--quick`` can be used to speed up the code generation.
It skips the generation of LaTeX-formatted equations, which are only used in documentation and the interactive
mode.

andes run
-------------
``andes run`` is the entry point for power system analysis routines.
``andes run`` takes one positional argument, ``filename`` , along with other optional keyword arguments.
``filename`` is the test case path, either relative or absolute.
Without other options, ANDES will run power flow calculation for the provided file.

``andes run`` has an option for the program verbosity level, controlled by ``-v`` or ``--verbose``.
Accpeted levels are the same as in the ``logging`` module: 10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR,
50 - CRITICAL.
To show debugging outputs, use ``-v 10``.

Power flow
..........

To perform a power flow study for test case named ``kundur_full.xlsx`` in the current directory, run

.. code:: bash

    andes run kundur_full.xlsx

The full path to the case file is also accepted, for example,

.. code:: bash

    andes run /home/user/andes/cases/kundur/kundur_full.xlsx

Power flow reports will be saved to the current directory in which andes is called.
The power flow report contains four sections: a) system statistics, b) ac bus
and dc node data, c) ac line data, and d) the initialized values of other
algebraic variables and state variables.

Time Domain Simulation
......................

To run the time domain simulation (TDS) for ``kundur_full.xlsx``, run

.. code:: bash

    andes run kundur_full.xlsx -r tds

The output looks like::

    ANDES 0.6.8 (Git commit id 0ace2bc0, Python 3.7.6 on Darwin)
    Session: hcui7, 02/09/2020 10:35:37 PM

    Parsing input file </Users/hcui7/repos/andes/tests/kundur_full.xlsx>
    Input file kundur_full.xlsx parsed in 0.5425 second.
    -> Power flow calculation with Newton Raphson method:
    0: |F(x)| = 14.9283
    1: |F(x)| = 3.60859
    2: |F(x)| = 0.170093
    3: |F(x)| = 0.00203827
    4: |F(x)| = 3.76414e-07
    Converged in 5 iterations in 0.0080 second.
    Report saved to </Users/hcui7/repos/andes/tests/kundur_full_out.txt> in 0.0036 second.
    -> Time Domain Simulation:
    Initialization tests passed.
    Initialization successful in 0.0152 second.
      0%|                                                    | 0/100 [00:00<?, ?%/s]
      <Toggle 0>: Applying status toggle on Line idx=Line_8
    100%|██████████████████████████████████████████| 100/100 [00:03<00:00, 28.99%/s]
    Simulation completed in 3.4500 seconds.
    TDS outputs saved in 0.0377 second.
    -> Single process finished in 4.4310 seconds.

This execution first solves the power flow as a starting point.
Next, the numerical integration simulates 20 seconds, during which a predefined
breaker opensat 2 seconds.

TDS produces two output files by default: a NumPy data file ``ieee14_syn_out.npy``
and a variable name list file ``ieee14_syn_out.lst``.
The list file contains three columns: variable indices, variabla name in plain text, and variable
name in LaTeX format.
The variable indices are needed to plot the needed variable.

Multiprocess
............
ANDES takes multiple files inputs or wildcard.
Multiprocessing will be triggered if more than one valid input files are found.
For example, to run power flow for files with a prefix of ``case5`` and a suffix (file extension)
of ``.m``, run

.. code:: bash

    andes run case5*.m

Test cases that match the pattern, including ``case5.m`` and ``case57.m``, will be processed.

Routine
.......
Option ``-r`` or ``-routine`` is used for specifying the analysis routine, followed by the routine name.
Available routine names include ``pflow, tds, eig``.
`pflow` for power flow, `tds` for time domain simulation, and `eig` for eigenvalue analysis.
`pflow` is default even if ``-r`` is not given.

For example, to run time-domain simulation for ``kundur_full.xlsx`` in the current
directory, run

.. code:: bash

    andes run kundur_full.xlsx -r tds

Convert
.......
ANDES recognizes a few input formats and can convert input systems into the ``xlsx`` format.
This function is useful when one wants to use models that are unique in ANDES.

The command for converting is ``--convert``, following the output format (only ``xlsx`` is currently supported).
For example, to convert ``case5.m`` into the ``xlsx`` format, run

.. code:: bash

    andes run case5.m --convert xlsx

The output will look like ::

    ANDES 0.6.8 (Git commit id 0ace2bc0, Python 3.7.6 on Darwin)
    Session: hcui7, 02/09/2020 10:22:14 PM

    Parsing input file </Users/hcui7/repos/andes/cases/matpower/case5.m>
    CASE5  Power flow data for modified 5 bus, 5 gen case based on PJM 5-bus system
    Input file case5.m parsed in 0.0033 second.
    xlsx file written to </Users/hcui7/repos/andes/cases/matpower/case5.xlsx>
    Converted file /Users/hcui7/repos/andes/cases/matpower/case5.xlsx written in 0.5079 second.
    -> Single process finished in 0.8765 second.

Note that ``--convert`` will only create sheets for existing models.
In case one want to create template sheets to add models later, ``--convertall`` can be used.

andes plot
--------------
``andes plot`` is the command-line tool for plotting.
It currently supports time-domain simulation data.
Three positional arguments are required, and a dozen of optional arguments are supported.

positional arguments:
  ========              =====================
  Argument              Description
  --------              ---------------------
  datfile               dat file name.
  x                     x axis variable index
  y                     y axis variable index
  ========              =====================

optional arguments:
  ==========================    ======================================
  Argument                      Description
  --------------------------    --------------------------------------
  -h, --help                    show this help message and exit
  --xmin LEFT                   x axis minimum value
  --xmax RIGHT                  x axis maximum value
  --ymax YMAX                   y axis maximum value
  --ymin YMIN                   y axis minimum value
  --checkinit                   check initialization value
  -x XLABEL, --xlabel XLABEL
                                manual x-axis text label
  -y YLABEL, --ylabel YLABEL
                                y-axis text label
  -s, --save                    save to file
  -g, --grid                    grid on
  -d, --no_latex                disable LaTex formatting
  -n, --no_show                 do not show the plot window
  --ytimes YTIMES               y times
  --dpi DPI                     image resolution in dot per inch (DPI)
  ==========================    ======================================

For example, to plot the generator speed variable of synchronous generator 1
``omega Syn 1`` versus time, read the indices of the variable (44) and time
(0), run

.. code:: bash

    andes plot ieee14_syn_out.npy 0 44

In this command, ``ande splot`` is a plotting tool for TDS output files.
``ieee14_syn_out.npy`` is data file name. ``0`` is the index of ``Time`` for
the x-axis. ``44`` is the index of ``omega Syn 1``.

The y-axis variabla indices can also be specified in the Python range fashion
. For example, ``andes plot ieee14_syn_out.npy 0 44:69:6`` will plot the
variables at indices 44, 50, 56, 62, and 68.

``andes plot`` will attempt to render the image with LaTeX if ``dvipng``
program is in the search path. In case LaTeX is available but fails (happens
on Windows), the option ``-d`` can be used to disable LaTeX rendering.

andes misc
--------------
``andes misc`` contains miscellaneous functions, such as configuration and output cleaning.

``--save-config``

Saves all configs to a file. By default, saves to ``~/.andes/andes.conf`` file.

This file contains all the runtime configs for the system and routines.

``--edit-config``

You can change the configuration of ANDES run by saving the config and editing it.

Run ``andes misc --save-config`` to save the config file to the default location.
Then, run ``andes misc --edit-config`` to edit it. On Microsoft Windows, it will
open up a notepad. On Linux, it will use the ``$EDITOR`` environment variable
or use ``gedit`` by default. On macOS, the default is vim.

``-C, --clean``

Option to remove any generated files. Removes files with any of the following
suffix: ``_out.txt`` (power flow report), ``_out.dat`` (time domain data),
``_out.lst`` (time domain variable list), and ``_eig.txt`` (eigenvalue report).


Interactive Usage
=================

Running Studies
---------------

The Andes Python APIs are loaded into an interactive Python environment
(Python, IPython or Jupyter Notebook) using ``import``. To start, import the
whole package and set up the global logger using

    >>> import andes
    >>> andes.main.config_logger(log_file=None)

Create an instance of Power System from the case file, for example, at ``
ieee14_syn.dm``
whole package and set up the global logger using

    >>> import andes
    >>> andes.main.config_logger(log_file=None)

Create an instance of Power System from the case file, for example, at ``
ieee14_syn.dm``
whole package and set up the global logger using

    >>> import andes
    >>> andes.main.config_logger(logfile=None)

Create an instance of Power System from the case file, for example, at ``
ieee14_syn.dm`` ::

    >>> ps = andes.system.PowerSystem('ieee14_syn.dm')

Next, guess the input file format and parse the data into the system ::

    >>> andes.filters.guess(ps)
    'dome'
    >>> andes.filters.parse(ps)
    Parsing input file <ieee14_syn.dm>
    True

Next, set up the system structure using the parsed input data

    >>> ps.setup()
    <andes.system.PowerSystem at 0x7fd5ea96d4e0>

To continue, run the power flow study using

    >>> ps.pflow.run()
    -> Power flow study: NR method, non-flat start
    Iter 1.  max mismatch = 2.1699877
    Iter 2.  max mismatch = 0.2403104
    Iter 3.  max mismatch = 0.0009915
    Iter 4.  max mismatch = 0.0000001
    Solution converged in 0.0038 second in 4 iterations
    Out[8]: (True, 4)

To change the run config, change the attributes in ``ps.pflow.config``. The
config options can be printed out with ``print(ps.pflow.config.doc())``.

Before running the TDS or eigenvalue analysis, the dynamic components needs
to be initialized with

    >> ps.tds.init()

Run the next analysis routine, for example, TDS, with

    >>> ps.tds.run()
    -> Time Domain Simulation: trapezoidal method, t=20 s
    <Fault> Applying fault on Bus <4.0> at t=2.0.              |ETA:  0:00:00]
    <Fault> Clearing fault on Bus <4.0> at t=2.05.
    [100%|#####################################################|Time: 0:00:01]
    Time domain simulation finished in 1.2599 seconds.
    True

Save the results to list and data files with

    >>> ps.tds.dump_results()
    Simulation data sumped in 0.0978 seconds.


Plotting Results
----------------

The ``andes.plot`` package can be used interactively for plotting time-domain
simulation results. Import functions from the package using

    >>> from andes.plot import read_dat, read_label, do_plot

Specify the files and the indices to plot using

    >>> dat_file = 'ieee14_syn_out.dat'
    >>> lst_file = 'ieee14_syn_out.lst'
    >>> x_idx = [0]
    >>> y_idx = [44, 50, 56]

Call functions `read_dat` and `read_label` to read out the values and names based on the variable indices.

    >>> x_dat, y_dat = read_dat(dat_file, x_idx, y_idx)
    >>> x_name, y_name = read_label(lst_file, x_idx, y_idx)

Call function `do_plot` to plot the curves

    >>> fig, ax = do_plot(xdata=x_dat, ydata=y_dat, 
                          xname=x_name, yname=y_name, 
                          ylabel='Generator Speed [pu]', grid=True)


Pretty Print of Equations
----------------------------------------
Each ANDES models offers pretty print of LaTeX-formatted equations in the jupyter notebook environment.

To use this feature, symbolic equations need to be generated in the current session using ::

    import dill
    dill.settings['recurse'] = True

    import andes
    sys = andes.system.System()
    sys.prepare()

This process may take several seconds to complete. Once done, equations can be viewed by accessing
``sys.<ModelName>.<EquationName>_print``, where ``<ModelName>`` is the model name and ``<EquationAttr>`` is the
equation name.

Supported equation names include the following:

- ``f``: differential equations for states :math:`\textbf{f}=\dot{x}`
- ``g``: algebraic equations for algebraic variables :math:`\textbf{g}=0`
- ``df``: derivatives of ``f`` over all variables
- ``dg``: derivatives of ``g`` over all variables
- ``s`` the value equations for service variables

For example, to print the algebraic equations of model ``GENCLS``, one can use ``sys.GENCLS.g_print``.

In addition to equations, all variable symbols can be printed at ``sys.<ModelName>.vars_print``.


