.. _chap-walkthrough:

******************
Getting Started
******************
This chapter describes the basic usage of a) the command-line interface, b)
the Python functions for Notebook or IPython interactive environment.

.. _sec-command:

Command Line Usage
=======================

Command Line Options
--------------------

Andes is invoked from the command line using the command ``andes``. It prints
out a splash screen with version and environment information and exits with a
"no input" error. ::

    ANDES 0.5.5 (Build gdded710, Python 3.5.2 on Linux)
    Session: 09/05/2018 02:05:18 PM

    error: no input file. Try 'andes -h' for help.

Help for the command line options can be retrieved using the ``-h`` or
``--help`` command, namely, ``andes -h``.

Frequently Used Options
-----------------------

``casename``

One positional argument, path to the case file, is required for Andes to
populate the power system object and perform studies. The case file, however,
is omitted if any utility options are specified in the optional arguments.

To perform a power flow study of the file named ``ieee14.dm`` in the current
directory, run ::

    andes ieee14.dm

Andes also takes the full path to the case file, for example, ::

    andes /home/hcui7/andes/cases/ieee14.dm

Output files will be saved to the current directory where Andes is called.

Andes also takes multiple files or wildcard. Multiprocessing will be
triggered if more than one valid input files are found. For example, to run
power flow for files with a prefix of ``ieee1`` and a suffix (file extension)
of ``.dm``, run

    andes ieee1*.dm

Andes will run the case files that match the pattern such as ``ieee14.dm``
and ``ieee118.dm``.

``-r, -routine {pflow, tds, eig}``

Option for specifying the analysis routine. `pflow` for power flow, `tds` for
time domain simulation, and `eig` for eigenvalue analysis. `pflow` as the
default is this option is not given.

For example, to run time domain simulation for ``ieee14.dm`` in the current
directory, run

    andes ieee14.dm -r tds

``-C, --clean``

Option to remove any generated files. Removes files with any of the following
suffix: ``_out.txt`` (power flow report), ``_out.dat`` (time domain data),
``_out.lst`` (time domain variable list), and ``_eig.txt`` (eigenvalue report).

``-g, --group``

Option to print out all the models in a group. To print out all the groups
and models they contain, run ``andes -g all``.

``-q, --quick-help``

Print out a quick help of parameter definitions of a single given model. For
example, ``andes -q Bus`` prints out the parameter definition of the model
``Bus``.

Parameters with an asterisk ``*`` are mandatory. Parameters with a number
sign ``#`` are per unit values in the element base.

``--help-config``

Print out a table of help for the specified configurations. Available options
are ``all``, ``system`` or any routine name such as ``pflow`` and ``eig``.

For example, ``andes --help-config all`` prints out all the config helps.

``--save-config``

Saves all configs to a file. By default, save to ``~/.andes/andes.conf`` file.

This file contains all the runtime configs for the system and routines.

``--load-config``

Load an Andes config file that occurs first in the following search path: a)
the specified path, b) current directory, c) home directory

``-v, --verbose``
Verbosity level in (10, 20, 30, 40, 50) for (DEBUG, INFO, WARNING, ERROR,
CRITICAL). The default is 20 (INFO). Set to 10 for debugging.

Plotting Tool
-------------

Andes comes with a command-line plotting tool for time-domain simulation
output data.

Examples
--------

Power Flow
----------

The example test cases are in the ``cases`` folder of the package.

Run power flow for ``ieee14_syn.dm`` using the command ::

    $ andes ieee14_syn.dm
    ANDES 0.5.5 (Build g651fdac, Python 3.5.2 on Linux)
    Session: 09/06/2018 11:02:52 AM

    Parsing input file <ieee14_syn.dm>
    -> Power flow study: NR method, non-flat start
    Iter 1.  max mismatch = 2.1699877
    Iter 2.  max mismatch = 0.2403104
    Iter 3.  max mismatch = 0.0009915
    Iter 4.  max mismatch = 0.0000001
    Solution converged in 0.0028 second in 4 iterations
    report written to <ieee14_syn_out.txt> in 0.0014 second.
    -> Single process finished in 0.1537 second.

The printed message shows that the power flow uses the Newton Raphson (NR)
method with non-flat start. The solution process converges in four iterations
in 0.002 seconds. A report is written to the file <ieee14_syn_out.txt>.

The power flow report contains four sections: a) system statistics, b) ac bus
and dc node data, c) ac line data, and d) the initialized values of other
algebraic variables and state variables.


Change Run Config
-----------------

You can change the configuration of the power flow run by saving the config
and editing it.

Run ``andes --save-config`` to save the config file to the default location.
Then, run ``andes --edit-config`` to edit it. On Microsoft Windows, it will
open up a notepad. On Linux, it will use the ``$EDITOR`` environment variable
or use ``gedit`` by default. On macOS, the default is vim.

To change the power flow solution method, for example, from NR to Fast
Decoupled Power Flow (FDPF), find ``method = NR `` in the ``[Pflow]`` section
and modified it to

    method = FDPF

Note that FDPF is an available method. To view the available options, in a
command line window, run ``andes --help-config pflow``.

Time Domain Simulation
----------------------

To run the time domain simulation (TDS) for ``ieee14_syn.dm``, run ::

    $ andes ieee14.dm -r tds
    ANDES 0.5.5 (Build g651fdac, Python 3.5.2 on Linux)
    Session: 09/06/2018 11:18:55 AM

    Parsing input file <ieee14_syn.dm>
    -> Power flow study: NR method, non-flat start
    Iter 1.  max mismatch = 2.1699877
    Iter 2.  max mismatch = 0.2403104
    Iter 3.  max mismatch = 0.0009915
    Iter 4.  max mismatch = 0.0000001
    Solution converged in 0.0054 second in 4 iterations
    report written to <ieee14_syn_out.txt> in 0.0019 second.
    -> Time Domain Simulation: trapezoidal method, t=20 s
    <Fault> Applying fault on Bus <4.0> at t=2.0.
    <Fault> Clearing fault on Bus <4.0> at t=2.05.
    Time domain simulation finished in 1.2613 seconds.
    -> Single process finished in 1.3878 seconds.

This execution first solves the power flow as a starting point. Next, the
numerical integration is run to simulate 20 seconds during which a predefined
fault on Bus 4 happens at 2 seconds.

TDS produces two output files by default: a data file ``ieee14_syn_out.dat``
and a variable name list file ``ieee14_syn_out.lst``. The list file contains
three columns: variable indices, variabla name in plain text, and variable
name in LaTeX format. The variable indices are needed to plot the needed
variable.

Plottting the TDS Results
-------------------------

For example, to plot the generator speed variable of synchronous generator 1
``omega Syn 1`` versus time, read the indices of the variable (44) and time
(0), run ::

    andesplot ieee14_syn_out.dat 0 44

In this command, ``andesplot`` is a plotting tool for TDS output files.
``ieee14_syn_out.dat`` is data file name. ``0`` is the index of ``Time`` for
the x-axis. ``44`` is the index of ``omega Syn 1``.

The y-axis variabla indices can also be specified in the Python range fashion
. For example, ``andesplot ieee14_syn_out.dat 0 44:69:6`` will plot the
variables at indices 44, 50, 56, 62, and 68.

``andesplot`` will attempt to render the image with LaTeX if ``dvipng``
program is in the search path. In case LaTeX is available but fails (happens
on Windows), the option ``-d`` can be used to disable LaTeX rendering.

A complete list of options for ``andesplot`` is available using ``andesplot
-h``.

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


