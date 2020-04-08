.. _tutorial:

********
Tutorial
********
ANDES can be used as a command-line tool or a library.
The command-line interface (CLI) comes handy to run studies.
As a library, it can be used interactively in the IPython shell or the Jupyter Notebook.
This chapter describes the most common usages.

Please see the cheat sheet if you are looking for quick help.

.. _sec-command:

Command Line Usage
==================

Basic Usage
-----------

ANDES is invoked from the command line using the command ``andes``.
Running ``andes`` without any input is equal to  ``andes -h`` or ``andes --help``.
It prints out a preamble with version and environment information and help commands::

        _           _         | Version 0.8.4
       /_\  _ _  __| |___ ___ | Python 3.7.1 on Darwin, 04/07/2020 10:22:17 PM
      / _ \| ' \/ _` / -_|_-< |
     /_/ \_\_||_\__,_\___/__/ | This program comes with ABSOLUTELY NO WARRANTY.

    usage: andes [-h] [-v {10,20,30,40,50}]
                 {run,plot,misc,prepare,doc,selftest} ...

    positional arguments:
      {run,plot,misc,prepare,doc,selftest}
                            [run] run simulation routine; [plot] plot simulation
                            results; [doc] quick documentation; [prepare] run the
                            symbolic-to-numeric preparation; [misc] miscellaneous
                            functions.

    optional arguments:
      -h, --help            show this help message and exit
      -v {10,20,30,40,50}, --verbose {10,20,30,40,50}
                            Program logging level in 10-DEBUG, 20-INFO,
                            30-WARNING, 40-ERROR or 50-CRITICAL.

The first level of commands are chosen from ``{run,plot,misc,prepare,selftest}``. Each command contains a group
of subcommands, which can be looked up with ``-h``. For example, use ``andes run -h`` to look up the subcommands
in ``run``. The most commonly used commands will be explained in the following.

``andes`` has an option for the program verbosity level, controlled by ``-v`` or ``--verbose``.
Accepted levels are the same as in the ``logging`` module: 10 - DEBUG, 20 - INFO, 30 - WARNING, 40 - ERROR,
50 - CRITICAL.
To show debugging outputs, use ``-v 10``.

andes selftest
--------------
After installing ANDES, it is encouraged to use ``andes selftest`` to run tests and check the basic functionality.
It might take a minute to run the whole self-test suite. Results are printed as the tests proceed. An example
output looks like ::

    ANDES 0.8.1 (Git commit id gc954fc1, Python 3.7.3 on Linux)
    Session: hcui7, 03/22/2020 11:02:35 AM
    This program comes with ABSOLUTELY NO WARRANTY.

    test_docs (test_1st_system.TestCodegen) ... ok
    test_alter_param (test_case.Test5Bus) ... ok
    test_as_df (test_case.Test5Bus) ... ok
    test_cache_refresn (test_case.Test5Bus) ... ok
    test_count (test_case.Test5Bus) ... ok
    test_idx (test_case.Test5Bus) ... ok
    test_init_order (test_case.Test5Bus) ... ok
    test_names (test_case.Test5Bus) ... ok
    test_pflow (test_case.Test5Bus) ... ok
    test_pflow_reset (test_case.Test5Bus) ... ok
    test_tds_init (test_case.Test5Bus) ... ok
    test_eig_run (test_case.TestKundur2Area) ... ok
    test_tds_run (test_case.TestKundur2Area) ... ok
    test_npcc_raw (test_case.TestNPCCRAW) ... ok
    test_npcc_raw_convert (test_case.TestNPCCRAW) ... ok
    test_npcc_raw_tds (test_case.TestNPCCRAW) ... No dynamic component loaded.
    ok
    test_main_doc (test_cli.TestCLI) ... ok
    test_misc (test_cli.TestCLI) ... ok
    test_limiter (test_discrete.TestDiscrete) ... ok
    test_sorted_limiter (test_discrete.TestDiscrete) ... ok
    test_switcher (test_discrete.TestDiscrete) ... ok
    test_tree (test_paths.TestPaths) ... ok
    test_pflow_mpc (test_pflow_matpower.TestMATPOWER) ... ok

    ----------------------------------------------------------------------
    Ran 23 tests in 13.834s

    OK

Test cases can grow, and there could be more cases than above. Make sure that all tests have passed.

.. warning ::
    ANDES is getting updates frequently. After updating your copy, please run
    ``andes selftest`` to confirm the functionality. The command also makes sure the generated code is up to date.
    See `andes prepare`_ for more details on automatic code generation.

andes prepare
-----------------
.. _`andes prepare`:

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

Routine
.......
Option ``-r`` or ``-routine`` is used for specifying the analysis routine, followed by the routine name.
Available routine names include ``pflow, tds, eig``.
`pflow` for power flow, `tds` for time domain simulation, and `eig` for eigenvalue analysis.
`pflow` is default even if ``-r`` is not given.

For example, to run time-domain simulation for ``kundur_full.xlsx`` in the *current directory*, run

.. code:: bash

    andes run kundur_full.xlsx -r tds

The file is located at ``andes/cases/kundur/kundur_full.xlsx`` relative to the source code root folder.
Use ``cd`` to change directory to that folder on your machine.

Two output files, ``kundur_full_out.lst`` and ``kundur_full_out.npy`` will be created for variable names
and values, respectively.

Likewise, to run eigenvalue analysis for ``kundur_full.xlsx``, use

.. code:: bash

    andes run kundur_full.xlsx -r eig

The eigenvalue report will be written in a text file named ``kundur_full_eig.txt``.

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

Time-domain simulation
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

Disable output
..............
The output files can be disabled with option ``--no-output`` or ``-n``.
It is useful when only computation is needed without saving the results.

Profiling
.........
Profiling is useful for analyzing the computation time and code efficiency.
Option ``--profile`` enables the profiling of ANDES execution.
The profiling output will be written in two files in the current folder, one ending with ``_prof.txt`` and the
other one with ``_prof.prof``.

The text file can be opened with a text editor, and the ``.prof`` file can be visualized with ``snakeviz``,
which can be installed with ``pip install snakeviz``.

If the output is disabled, profiling results will be printed to stdio.

Multiprocessing
...............
ANDES takes multiple files inputs or wildcard.
Multiprocessing will be triggered if more than one valid input files are found.
For example, to run power flow for files with a prefix of ``case5`` and a suffix (file extension)
of ``.m``, run

.. code:: bash

    andes run case5*.m

Test cases that match the pattern, including ``case5.m`` and ``case57.m``, will be processed.

Option ``--ncpu NCPU`` can be used to specify the maximum number of parallel processes.
By default, all cores will be used. A small number can be specified to increase operation system responsiveness.

Format converter
................
.. _`format converter`:

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

In case one wants to create template sheets to add models later, ``--convert-all`` can be used instead.

If one wants to add workbooks to an existing xlsx file, use option ``--add-book ADD_BOOK``, where ``ADD_BOOK``
can be a single model name or comma-separated model names (without any space).

.. Warning::
    With ``--add-book``, the xlsx file will be overwritten.
    Any **empty or non-existent models** will be REMOVED.

andes plot
--------------
``andes plot`` is the command-line tool for plotting.
It currently supports time-domain simulation data.
Three positional arguments are required, and a dozen of optional arguments are supported.

positional arguments:
  ==============              ===========================
  Argument                    Description
  --------------              ---------------------------
  | filename                    simulation output file name, which should end with
  |                             `out`. File extension can be omitted.
  x                           the X-axis variable index, typically 0 for Time
  y                           Y-axis variable indices. Space-separated indices or a
                              colon-separated range is accepted
  ==============              ===========================

For example, to plot the generator speed variable of synchronous generator 1
``GENROU omega 0`` versus time, read the indices of the variable (2) and time
(0), run

.. code:: bash

    andes plot kundur_full_out.lst 0 2

In this command, ``andes plot`` is the plotting command for TDS output files.
``kundur_full_out.lst`` is list file name. ``0`` is the index of ``Time`` for
the x-axis. ``2`` is the index of ``GENROU omega 0``. Note that for the the file name,
either ``kundur_full_out.lst`` or ``kundur_full_out.npy`` works, as the program will
automatically extract the file name.

The y-axis variabla indices can also be specified in the Python range fashion
. For example, ``andes plot kundur_full_out.npy 0 2:21:6`` will plot the
variables at indices 2, 8, 14 and 20.

``andes plot`` will attempt to render with LaTeX if ``dvipng`` program is in the search path.
Figures rendered by LaTeX is considerably better in symbols quality but takes much longer time.
In case LaTeX is available but fails (frequently happens on Windows), the option ``-d`` can be used to disable
LaTeX rendering.

Other optional arguments are listed in the following.

optional arguments:
    ============================    ======================================================
    Argument                        Description
    ----------------------------    ------------------------------------------------------
    optional arguments:
    -h, --help                      show this help message and exit
    --xmin LEFT                     minimum value for X axis
    --xmax RIGHT                    maximum value for X axis
    --ymax YMAX                     maximum value for Y axis
    --ymin YMIN                     minimum value for Y axis
    --find FIND                     find variable indices that matches the given pattern
    | --xargs XARGS                 find variable indices and return as a list of
    |                               arguments usable with "| xargs andes plot"
    --exclude EXCLUDE               pattern to exclude in find or xargs results
    -x XLABEL, --xlabel XLABEL      x-axis label text
    -y YLABEL, --ylabel YLABEL      y-axis label text
    -s, --savefig                   save figure. The default fault is `png`.
    | -format SAVE_FORMAT           format for savefig. Common formats such as png, pdf, jpg are supported
    --dpi DPI                       image resolution in dot per inch (DPI)
    -g, --grid                      grid on
    --greyscale                     greyscale on
    -d, --no-latex                  disable LaTex formatting
    -n, --no-show                   do not show the plot window
    --ytimes YTIMES                 scale the y-axis values by YTIMES
    -c, --tocsv                     convert npy output to csv
    ============================    ======================================================

andes doc
---------
``andes doc`` is a tool for quick lookup of model documentation.
The basic usage of ``andes doc`` is to provide a model name as the positional argument.
It will print out model parameters, variables, and equations to the stdio.
If you are looking for full documentation, visit `andes.readthedocs.io <https://andes.readthedocs.io>`_.

It is intended as a quick way for documentation.
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


andes misc
----------
``andes misc`` contains miscellaneous functions, such as configuration and output cleaning.

Configuration
.............
ANDES uses a configuration file to set runtime configs for the system routines, and models.
``--save-config`` saves all configs to a file. By default, it saves to ``~/.andes/andes.conf`` file, where ``~``
is the path to your home directory.

With ``--edit-config``, you can edit ANDES configuration handy.
The command will automatically save the configuration to the default location if not exist.
The shorter version ``--edit`` can be used instead asn Python automatically matches it with ``--edit-config``.

You can pass an editor name to ``--edit``, such as ``--edit vim``.
If the editor name is not provided, it will use the following defaults:
- Microsoft Windows: notepad.
- GNU/Linux: the ``$EDITOR`` environment variable, or ``gedit`` if not exist.

For macOS users, the default is vim.
If not familiar with vim, you can use nano with ``--edit nano`` or TextEdit with
``--edit "open -a TextEdit"``.

Cleanup
.......
``-C, --clean``

Option to remove any generated files. Removes files with any of the following
suffix: ``_out.txt`` (power flow report), ``_out.dat`` (time domain data),
``_out.lst`` (time domain variable list), and ``_eig.txt`` (eigenvalue report).

Interactive Usage
=================
This section is a tutorial for using ANDES in an interactive environment.
All interactive shells are supported, including Python shell, IPython, Jupyter Notebook and Jupyter Lab.
The examples below uses Jupyter Notebook.

Jupyter Notebook
----------------
Jupyter notebook is used as an example. Jupyter notebook can be installed with

.. code:: bash

    conda install jupyter notebook

After the installation, change directory to the folder that you wish to store notebooks,
start the notebook with

.. code:: bash

    jupyter notebook

A browser window should open automatically with the notebook browser loaded.
To create a new notebook, use the "New" button at the top right corner.

Import
------
Like other Python libraries, ANDES can be imported into an interactive Python environment.

    >>> import andes
    >>> andes.config_logger()

The ``config_logger`` is needed to print logging information in the current session.
Otherwise, information messages will be silenced, and only warnings and error will be printed.

To enable debug messages, use

    >>> andes.config_logger(stream_level=10)

If you have not run ``andes prepare``, use the command once to generate code

    >>> andes.prepare()


Create Test System
------------------
Before running studies, a "System" object needs to be create to hold the system data.
The System object can be created by passing the path to the case file the entrypoint function.
For example, to run the file ``kundur_full.xlsx`` in the same directory as the notebook, use

    >>> ss = andes.run('kundur_full.xlsx')

This function will parse the input file, run the power flow, and return the system as an object.
Outputs will look like ::

    Parsing input file </Users/hcui7/notebooks/kundur/kundur_full.xlsx>
    Input file kundur_full.xlsx parsed in 0.4172 second.
    -> Power flow calculation with Newton Raphson method:
    0: |F(x)| = 14.9283
    1: |F(x)| = 3.60859
    2: |F(x)| = 0.170093
    3: |F(x)| = 0.00203827
    4: |F(x)| = 3.76414e-07
    Converged in 5 iterations in 0.0222 second.
    Report saved to </Users/hcui7/notebooks/kundur_full_out.txt> in 0.0015 second.
    -> Single process finished in 0.4677 second.

In this example, ``ss`` is an instance of ``andes.System``.
It contains member attributes for models, routines, and numerical DAE.

Naming convention for the ``System`` attributes are as follows

- Model attributes share the same name as class names. For example, ``ss.Bus`` is the ``Bus`` instance.
- Routine attributes share the same name as class names. For example, ``ss.PFlow`` and ``ss.TDS`` are the
  routine instances.
- The numerical DAE instance is in lower case ``ss.dae``.

Inspect Parameter
--------------------
Parameters for the loaded system can be easily inspected in Jupyter Notebook using Pandas.

Input parameters for each model instance is in the ``cache.df_in`` attribute.
For example, to view the input parameters for ``Bus``, use ::

    >>> ss.Bus.cache.df_in

A table will be printed with the columns being each parameter and the rows being Bus instances.
Parameter in the table is the same as the input file without per-unit conversion.

Parameters are converted to per unit values under system base.
To view the per unit values, use the ``cache.df`` attribute.
For example, to view the system-base per unit value of ``GENROU``, use ::

    >>> ss.GENROU.cache.df

Running Studies
---------------

Three routines are currently supported: PFlow, TDS and EIG.
Each routine provides a ``run()`` method to execute.
The System instance contains member attributes having the same names.
For example, to run the time-domain simulation for ``ss``, use ::

    >>> ss.TDS.run()

Plotting TDS Results
--------------------
TDS comes with a plotting utility for interactive usage.
After running the simulation, a ``plotter`` attributed will be created for ``TDS``.
To use the plotter, provide the attribute instance of the variable to plot.
For example, to plot all the generator speed, use ::

    >>> ss.TDS.plotter.plot(ss.GENROU.omega)

Optional indices is accepted to choose the specific elements to plot.
It can be passed as a tuple to the ``a`` argument ::

    >>> ss.TDS.plotter.plot(ss.GENROU.omega, a=(0, ))

In the above example, the speed of the "zero-th" generator will be plotted.

Scaling
.......
A lambda function can be passed to argument ``ycalc`` to scale the values.
This is useful to convert a per-unit variable to nominal.
For example, to plot generator speed in Hertz, use ::

    >>> ss.TDS.plotter.plot(ss.GENROU.omega, a=(0, ),
                            ycalc=lambda x: 60*x,
                            )

Formatting
..........
A few formatting arguments are supported:

- ``grid = True`` to turn on grid display
- ``greyscale = True`` to switch to greyscale
- ``ylabel`` takes a string for the y-axis label

Pretty Print of Equations
----------------------------------------
Each ANDES models offers pretty print of LaTeX-formatted equations in the jupyter notebook environment.

To use this feature, symbolic equations need to be generated in the current session using ::

    import andes
    ss = andes.System()
    ss.prepare()

This process may take several seconds to complete. Once done, equations can be viewed by accessing
``ss.<ModelName>.<EquationName>_print``, where ``<ModelName>`` is the model name and ``<EquationName>`` is the
equation name.

.. Note ::

    Pretty print only works for the particular System instance whose ``prepare()`` method is called.
    In the above example, pretty print only works for ``ss`` after calling ``prepare()``.

Supported equation names include the following:

- ``f``: differential equations for states :math:`\textbf{f}=\dot{x}`
- ``g``: algebraic equations for algebraic variables :math:`\textbf{g}=0`
- ``df``: derivatives of ``f`` over all variables
- ``dg``: derivatives of ``g`` over all variables
- ``s`` the value equations for service variables

For example, to print the algebraic equations of model ``GENCLS``, one can use ``ss.GENCLS.g_print``.

In addition to equations, all variable symbols can be printed at ``ss.<ModelName>.vars_print``.

.. _formats:

I/O Formats
===========

Input Formats
-------------

ANDES currently supports the following input formats:

- ANDES Excel (.xlsx)
- MATPOWER (.m)
- PSS/E RAW (.raw)
- PSS/E DYR (.dyr), work in progress


ANDES xlsx Format
-----------------

The ANDES xlsx format is a newly introduced format since v0.8.0.
This format uses Microsoft Excel for conveniently viewing and editing model parameters.
You can use `LibreOffice <https://www.libreoffice.org>`_ or `WPS Office <https://www.wps.com/>`_ alternatively to
Microsoft Excel.

xlsx Format Definition
......................

The ANDES xlsx format contains multiple workbooks (tabs at the bottom).
Each workbook contains the parameters of all instances of the model, whose name is the workbook name.
The first row in a worksheet is used for the names of parameters available to the model.
Starting from the second row, each row corresponds to an instance with the parameters in the corresponding columns.
An example of the ``Bus`` workbook is shown in the following.

.. image:: images/tutorial/xlsx-bus.png
   :width: 600
   :alt: Example workbook for Bus

A few columns are used across all models, including ``uid``, ``idx``, ``name`` and ``u``.

- ``uid`` is an internally generated unique instance index. This column can be left empty if the xlsx file is
  being manually created. Exporting the xlsx file with ``--convert`` will automatically assign the ``uid``.
- ``idx`` is the unique instance index for referencing. An unique ``idx`` should be provided explicitly for each
  instance. Accepted types for ``idx`` include numbers and strings without spaces.
- ``name`` is the instance name.
- ``u`` is the connectivity status of the instance. Accepted values are 0 and 1. Unexpected behaviors may occur
  if other numerical values are assigned.

As mentioned above, ``idx`` is the unique index for an instance to be referenced.
For example, a PQ instance can reference a Bus instance so that the PQ is connected to the Bus.
This is done through providing the ``idx`` of the desired bus as the ``bus`` parameter of the PQ.

.. image:: images/tutorial/xlsx-pq.png
   :width: 600
   :alt: Example workbook for PQ

In the example PQ workbook shown above, there are two PQ instances on buses with ``idx`` being 7 and 8,
respectively.

Convert to xlsx
...............
Please refer to the the ``--convert`` command for converting a recognized file to xlsx.
See `format converter`_ for more detail.

Data Consistency
................

Input data needs to have consistent types for ``idx``. Both string and numerical types are allowed
for ``idx``, but the original type and the referencing type must be the same. For example,
suppose we have a bus and a connected PQ.
The Bus device may use ``1`` or ``'1'`` as its ``idx``, as long as the
PQ device uses the same value for its ``bus`` parameter.


The ANDES xlsx reader will try to convert data into numerical types when possible.
This is especially relevant when the input ``idx`` is string literal of numbers,
the exported file will have them converted to numbers.
The conversion does not affect the consistency of data.

Parameter Check
...............
The following parameter checks are applied after converting input values to array:

- Any ``NaN`` values will raise a ``ValueError``
- Any ``inf`` will be replaced with :math:`10^{8}`, and ``-inf`` will be replaced with :math:`-10^{8}`.


Cheatsheet
===========
A cheatsheet is available for quick lookup of supported commands.

View the PDF version at

https://www.cheatography.com//cuihantao/cheat-sheets/andes-for-power-system-simulation/pdf/

Make Documentation
==================

The documentation can be made locally into a variety of formats.
To make HTML documentation, change directory to ``docs``, and do

.. code:: bash

    make html

After a minute, HTML documentation will be saved to ``docs/build/html`` with the index page being ``index.html``.

A list of supported formats is as follows. Note that some format require additional compiler or library ::

    html        to make standalone HTML files
    dirhtml     to make HTML files named index.html in directories
    singlehtml  to make a single large HTML file
    pickle      to make pickle files
    json        to make JSON files
    htmlhelp    to make HTML files and an HTML help project
    qthelp      to make HTML files and a qthelp project
    devhelp     to make HTML files and a Devhelp project
    epub        to make an epub
    latex       to make LaTeX files, you can set PAPER=a4 or PAPER=letter
    latexpdf    to make LaTeX and PDF files (default pdflatex)
    latexpdfja  to make LaTeX files and run them through platex/dvipdfmx
    text        to make text files
    man         to make manual pages
    texinfo     to make Texinfo files
    info        to make Texinfo files and run them through makeinfo
    gettext     to make PO message catalogs
    changes     to make an overview of all changed/added/deprecated items
    xml         to make Docutils-native XML files
    pseudoxml   to make pseudoxml-XML files for display purposes
    linkcheck   to check all external links for integrity
    doctest     to run all doctests embedded in the documentation (if enabled)
    coverage    to run coverage check of the documentation (if enabled)
