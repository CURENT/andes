
.. _sec-command:

Command Line Usage
==================

Basic Usage
-----------

ANDES is invoked from the command line using the command ``andes``.
Running ``andes`` without any input is equal to  ``andes -h`` or ``andes --help``.
It prints out a preamble with version and environment information and help commands::

        _           _         | Version 1.3.4
       /_\  _ _  __| |___ ___ | Python 3.8.6 on Linux, 03/17/2021 11:28:55 AM
      / _ \| ' \/ _` / -_|_-< |
     /_/ \_\_||_\__,_\___/__/ | This program comes with ABSOLUTELY NO WARRANTY.

    usage: andes [-h] [-v {1,10,20,30,40}]
            {run,plot,doc,misc,prepare,selftest} ...

    positional arguments:
    {run,plot,doc,misc,prepare,selftest}
                        [run] run simulation routine; [plot] plot results;
                        [doc] quick documentation; [misc] misc. functions;
                        [prepare] prepare the numerical code; [selftest] run
                        self test.

    optional arguments:
    -h, --help            show this help message and exit
    -v {1,10,20,30,40}, --verbose {1,10,20,30,40}
                        Verbosity level in 10-DEBUG, 20-INFO, 30-WARNING, or
                        40-ERROR.

.. note::

    If the ``andes`` command is not found, check if (1) the installation was successful, and
    (2) you have activated the environment where ANDES is installed.

The first-level commands are chosen from ``{run,plot,doc,misc,prepare,selftest}``.
Each command contains a group of sub-commands, which can be looked up with ``-h``.
For example, use ``andes run -h`` to look up the sub-commands for ``run``.
The most frequently used commands are explained in the following.

``andes`` has an option for the program verbosity level, controlled by ``-v LEVEL`` or ``--verbose LEVEL``,
where level is a number chosen from the following:
1 (DEBUG with code location info), 10 (DEBUG), 20 (INFO), 30 (WARNING), 40 (ERROR), or 50 (CRITICAL).
For example, to show debugging outputs, use ``andes -v 10``, followed by the first-level commands.
The default logging level is 20 (INFO).

andes selftest
--------------
After the installation, please run ``andes selftest`` from the command line to test ANDES functionality.
It might take a minute to run the full self-test suite.
An example output looks like ::

    test_docs (test_1st_system.TestCodegen) ... ok
    test_alter_param (test_case.Test5Bus) ... ok
    ...
    ... (outputs are truncated)
    ...
    test_pflow_mpc (test_pflow_matpower.TestMATPOWER) ... ok

    ----------------------------------------------------------------------
    Ran 23 tests in 13.834s

    OK

There may be more test than what is shown above. Make sure that all tests have passed.

.. warning ::
    ANDES is getting updates frequently. After every update, please run
    ``andes selftest`` to confirm the functionality.
    The command also makes sure the generated code is up to date.
    See `andes prepare`_ for more details on automatic code generation.

andes prepare
-----------------
.. _`andes prepare`:

The symbolically defined models in ANDES need to be generated into numerical code for simulation.
The code generation can be manually called with ``andes prepare``.
Generated code are serialized to ``~/.andes/calls.pkl`` and dumped as Python code to ``~/.andes/pycode``.
In addition, ``andes selftest`` implicitly calls the code generation.
If you are using ANDES as a package in the user mode (namely, you have not modified or updated ANDES code),
you will not need to call it again.

.. note ::
    To developers:
    As of version 1.3.0, ANDES stores all generated Python code explicitly
    in ``.py`` files under the folder ``~/.andes/pycode``.
    Priority is given to Python code when reloading for simulation.

Option ``-q`` or ``--quick`` (enabled by default) can be used to speed up the code generation.
It skips the generation of :math:`\LaTeX`-formatted equations, which are only used in documentation and the interactive
mode.

Option ``-i`` or ``--incremental``, instead of ``-q``, can be used to further speed up the code generation
during model development.
``andes prepare -i`` only generates code for models that have been modified since the last code generation.

.. note ::
    To developers:
    ``andes prepare -i`` needs to be called immediately following any model equation modification.
    Otherwise, simulation results will not reflect the new equations and will likely lead to an error.

andes run
-------------
``andes run`` is the entry point for power system analysis routines.
``andes run`` takes one positional argument, ``filename`` , along with other optional keyword arguments.
``filename`` is the test case path, either relative or absolute.

For example, the command ``andes run kundur_full.xlsx`` uses a relative path.
If will work only if ``kundur_full.xlsx`` exists in the current directory of the command line.
The commands ``andes run /Users/hcui7/kundur_full.xlsx`` (on macOS) or
``andes run C:/Users/hcui7/kundur_full.xlsx`` (on Windows) use absolute paths to the case files
and do not depend on the command-line current directory.

.. note ::
    When working with the command line, use ``cd`` to change directory to the folder
    containing your test case.
    Spaces in folder and file names need to be escaped properly.

Routine
.......
Option ``-r`` or ``-routine`` is used for specifying the analysis routine,
followed by the routine name.
Available routine names include ``pflow, tds, eig``:
- ``pflow`` for power flow
- ``tds`` for time domain simulation
- ``eig`` for eigenvalue analysis

``pflow`` is the default if ``-r`` is not given.

Power flow
..........
Locate the ``kundur_full.xlsx`` file at ``andes/cases/kundur/kundur_full.xlsx`` under the source code folder,
or download it from
`the repository <https://github.com/cuihantao/andes/raw/master/andes/cases/kundur/kundur_full.xlsx>`_.

Change to the directory containing ``kundur_full.xlsx``.
To run power flow, execute the following in the command line:

.. code:: bash

    andes run kundur_full.xlsx

The full path to the case file is also recognizable, for example,

.. code:: bash

    andes run /home/user/andes/cases/kundur/kundur_full.xlsx

The power flow report will be saved to the current directory where ANDES is run.
The report contains four sections: a) system statistics, b) ac bus
and dc node data, c) ac line data, and d) the initialized values of other
algebraic variables and state variables.

Time-domain simulation
......................

To run the time domain simulation (TDS) for ``kundur_full.xlsx``, run

.. code:: bash

    andes run kundur_full.xlsx -r tds

The output looks like::

    Parsing input file </Users/user/repos/andes/tests/kundur_full.xlsx>
    Input file kundur_full.xlsx parsed in 0.5425 second.
    -> Power flow calculation with Newton Raphson method:
    0: |F(x)| = 14.9283
    1: |F(x)| = 3.60859
    2: |F(x)| = 0.170093
    3: |F(x)| = 0.00203827
    4: |F(x)| = 3.76414e-07
    Converged in 5 iterations in 0.0080 second.
    Report saved to </Users/user/repos/andes/tests/kundur_full_out.txt> in 0.0036 second.
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
breaker opens at 2 seconds.

TDS produces two output files by default:
a compressed NumPy data file ``kundur_full_out.npz``
and a variable name list file ``kundur_full_out.lst``.
The list file contains three columns:
variable indices, variable name in plain text, and variable
name in the :math:`\LaTeX` format.
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

The command for converting is ``--convert`` (or ``-c``),
following the output format (only ``xlsx`` is currently supported).
For example, to convert ``case5.m`` into the ``xlsx`` format, run

.. code:: bash

    andes run case5.m --convert xlsx

The output messages will look like ::

    Parsing input file </Users/user/repos/andes/cases/matpower/case5.m>
    CASE5  Power flow data for modified 5 bus, 5 gen case based on PJM 5-bus system
    Input file case5.m parsed in 0.0033 second.
    xlsx file written to </Users/user/repos/andes/cases/matpower/case5.xlsx>
    Converted file /Users/user/repos/andes/cases/matpower/case5.xlsx written in 0.5079 second.
    -> Single process finished in 0.8765 second.

Note that ``--convert`` will only create sheets for existing models.

In case one wants to create template sheets to add models later, ``--convert-all`` can be used instead.

If one wants to add workbooks to an existing xlsx file,
one can combine option ``--add-book ADD_BOOK`` (or ``-b ADD_BOOK``),
where ``ADD_BOOK`` can be a single model name or comma-separated
model names (without any space). For example,

.. code:: bash

    andes run kundur.raw -c -b Toggler

will convert file ``kundur.raw`` into an ANDES xlsx file (kundur.xlsx) and add
a template workbook for `Toggler`.

.. Warning::
    With ``--add-book``, the xlsx file will be overwritten.
    Any **empty or non-existent models** will be REMOVED.

PSS/E inputs
............
To work with PSS/E input files (.raw and .dyr), one need to provide the
raw file through ``casefile`` and pass the dyr file through ``--addfile``.
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

To create add a disturbance, there are two options.
The recommended option is to convert the PSS/E data into an ANDES xlsx file,
edit it and run (see the previous subsection).

An alternative is to edit the ``.dyr`` file with a planin-text editor (such as Notepad)
and append lines customized for ANDES models.
This is for advanced users after referring to ``andes/io/psse-dyr.yaml``,
at the end of which one can find the format of ``Toggler``: ::

    # === Custom Models ===
    Toggler:
        inputs:
            - model
            - dev
            - t

To define two Togglers in the ``.dyr`` file, one can append lines to the end
of the file using, for example, ::

    Line   'Toggler'  Line_2  1 /
    Line   'Toggler'  Line_2  1.1 /

which is separated by spaces and ended with a slash. The second parameter
is fixed to the model name quoted by a pair of single quotation marks,
and the others correspond to the fields defined in the above``inputs``.
Each entry is properly terminated with a forward slash.

andes plot
--------------
``andes plot`` is the command-line tool for plotting.
It currently supports time-domain simulation data.
Three positional arguments are required, and a dozen of optional arguments are supported.

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

For example, to plot the generator speed variable of synchronous generator 1
``omega GENROU 0`` versus time, read the indices of the variable (2) and time
(0), run

.. code:: bash

    andes plot kundur_full_out.lst 0 2

In this command, ``andes plot`` is the plotting command for TDS output files.
``kundur_full_out.lst`` is list file name. ``0`` is the index of ``Time`` for
the x-axis. ``2`` is the index of ``omega GENROU 0``. Note that for the the file name,
either ``kundur_full_out.lst`` or ``kundur_full_out.npy`` works, as the program will
automatically extract the file name.

The y-axis variabla indices can also be specified in the Python range fashion
. For example, ``andes plot kundur_full_out.npy 0 2:21:6`` will plot the
variables at indices 2, 8, 14 and 20.

``andes plot`` will attempt to render with :math:`\LaTeX` if ``dvipng`` program is in the search path.
Figures rendered by :math:`\LaTeX` is considerably better in symbols quality but takes much longer time.
In case :math:`\LaTeX` is available but fails (frequently happens on Windows), the option ``-d`` can be used to disable
:math:`\LaTeX` rendering.

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
    ----------------------------    ------------------------------------------------------
    --xargs XARGS                   find variable indices and return as a list of
                                    arguments usable with "| xargs andes plot"
    ----------------------------    ------------------------------------------------------
    --exclude EXCLUDE               pattern to exclude in find or xargs results
    -x XLABEL, --xlabel XLABEL      x-axis label text
    -y YLABEL, --ylabel YLABEL      y-axis label text
    -s, --savefig                   save figure. The default fault is `png`.
    ----------------------------    ------------------------------------------------------
    -format SAVE_FORMAT             format for savefig. Common formats such as png, pdf, jpg are supported
    ----------------------------    ------------------------------------------------------
    --dpi DPI                       image resolution in dot per inch (DPI)
    -g, --grid                      grid on
    --greyscale                     greyscale on
    -d, --no-latex                  disable LaTeX formatting
    -n, --no-show                   do not show the plot window
    --ytimes YTIMES                 scale the y-axis values by YTIMES
    -c, --to-csv                    convert npy output to csv
    ============================    ======================================================

.. _andes_doc:

andes doc
---------
``andes doc`` is a tool for quick lookup of model and routine documentation.
It is intended as a quick way for documentation.

The basic usage of ``andes doc`` is to provide a model name or a routine name as the positional argument.
For a model, it will print out model parameters, variables, and equations to the stdio.
For a routine, it will print out fields in the Config file.
If you are looking for full documentation, visit `andes.readthedocs.io <https://andes.readthedocs.io>`_.

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


andes misc
----------
``andes misc`` contains miscellaneous functions, such as configuration and output cleaning.

Configuration
.............
ANDES uses a configuration file to set runtime configs for the system routines, and models.
``andes misc --save-config`` saves all configs to a file.
By default, it saves to ``~/.andes/andes.conf`` file, where ``~``
is the path to your home directory.

With ``andes misc --edit-config``, you can edit ANDES configuration handy.
The command will automatically save the configuration to the default location if not exist.
The shorter version ``--edit`` can be used instead as Python matches it with ``--edit-config``.

You can pass an editor name to ``--edit``, such as ``--edit vim``.
If the editor name is not provided, it will use the following defaults:
- Microsoft Windows: notepad.
- GNU/Linux: the ``$EDITOR`` environment variable, or ``vim`` if not exist.

For macOS users, the default is vim.
If not familiar with vim, you can use nano with ``--edit nano`` or TextEdit with
``--edit "open -a TextEdit"``.

Cleanup
.......
``andes misc -C, --clean``

Option to remove any generated files. Removes files with any of the following
suffix: ``_out.txt`` (power flow report), ``_out.npy`` (time domain data),
``_out.lst`` (time domain variable list), and ``_eig.txt`` (eigenvalue report).
