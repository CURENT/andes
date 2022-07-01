.. _ReleaseNotes:

=============
Release notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

v1.7 Notes
==========

v1.7.4 (2022-07-01)
-------------------

- Renamed model `Toggler` to `Toggle`.
- New model: ESAC5A. (Contributed by Ahmad Ali).
- Added documentation for creating disturbances.
- Updated documentation for modeling blocks.

v1.7.3 (2022-06-25)
-------------------

Bug fix:

- Fix `Ipcmd` initialization equation of `REGCA1`.

Improved the interface to pandapower:

- Improved ``to_pandapower`` performance using vectorized conversion.
- Enhanced ``make_link_table`` to include group ``RenGen``.

v1.7.2 (2022-06-07)
-------------------
- Improved documentation and examples.

v1.7.1 (2022-05-31)
-------------------
This release contains minor fixes to the documentation.

Other changes:

- In ``PVD1``, Enable `pmx` limiting by default.
- In ``ST2CUT``, fix the type of ``busr2`` and ``busr`` to ``IdxParam``.

v1.7.0 (2022-05-22)
-------------------
Allow incrementally offloading simulation data from memory to the output file:

- ``[TDS].limit_store`` is a boolean value to enable the limit for in-memory
  time-series storage. If set to ``1``, data will be offloaded to the ``npz``
  file every ``[TDS].max_store`` steps. Offloaded data will then be erased from
  memory.
- If you need to interact with the time-series data in memory, you need to keep
  ``[TDS].limit_store`` to ``0``.

Allow specifying models, variables, and/or devices to output:

- See :ref:`Output`. The `model` field is mandatory. Leaving `varname` or `dev`
  blank indicates the selection of all applicable elements. For example,
  specifying `model` and `varname` without `dev` means that the variable for all
  devices will be exported.
- Plot tool works with in-memory time-series data specified by Output.

Simulation output control:

- Allow controlling the save frequency for output data in ``[TDS].save_every``.
  The default value is ``1``, which means that every step will be saved. Setting
  it to ``4``, for example, will save data every four steps. This setting
  applies to the in-memory storage and the output data file.
- Setting ``save_every = 0`` will immediately discard all data after each
  simulation step.
- Added the option ``[TDS].save_mode`` to change the automatic simulation data
  dumping to manual. Accepted values are ``auto`` and ``manual``. This option
  shall only be adjusted to ``manual`` when one is manually stepping the
  simulation and wants to avoid writing to the output file when the simulation
  reaches ``TDS.config.tf``. One will need to call ``TDS.save_output()`` when
  the full simulation concludes to avoid losing unsaved data.

Other changes:

- Fix the initialization of offline synchronous generators.
- Allow styles to be set for plots using the argument ``style``. To generate
  figures for IEEE publications, use ``style=ieee`` (require package
  ``scienceplots``).
- Moved the writing of the ``lst`` file to the first step of simulation.
- ``andes misc -C`` will not remove ``_out.csv`` file as it is considered data
  for post-processing just like exported figures.

v1.6 Notes
==========

v1.6.6 (2022-04-30)
-------------------
- Rename ``[System] call_stats`` to ``[System] save_stats`` for clarity. If
  turned on, one can retrieve statistics of function calls in
  ``TDS.call_stats``.
- Store routine execution time to routine member ``execution_time``.
- Fix PSS/E parsing issues with :ref:`GAST`.
- Fix issues and update default parameters for `REGCV1` and `REGCV2`.
- Allow adjusting limits for state variables during initialization. Like for
  algebraic variables, the default setting automatically adjusts the upper limit
  but not the lower one.

v1.6.5 (2022-04-19)
-------------------
- Added a TDS stop criteria based on rotor angle separation with reference to
  Power System Analysis Toolbox.
- Fix a bug for snapshot save and load. It now supports writing to and reading
  from ``io.BytesIO()``.

v1.6.4 (2022-04-17)
-------------------
Breaking change:

- ``PV`` model no longer has ``p`` as a variable in the DAE. ``p`` copies the
  value of ``p0``. This change affects the addresses of variables.
- Changed ``models.file_classes`` to a list to improve the control over
  the class initialization sequence in the same package.

Operator splitting for internal algebraic variables:

- ``VarService`` can be evaluated model-internal algebraic variables outside the
  DAE system. This approach is known as operator splitting and is commonly used
  in other simulation tools.
- Operator splitting reduces the size of the DAE system but introduces a
  one-iteration lag between the internal algebraic variables and others in the
  DAE system.
- ``VarService`` shall be avoided for singular functions (non-continuous) and
  shall not be adopted to circumvent initializing algebraic equations.
- ``VarService`` takes an argument ``sequential``, which is ``True`` by default.
  Non-sequential ``VarService`` shall not depend on other ``VarService``
  calculated at the same step as they will be evaluated simultaneously.
- :py:func:`andes.interop.pandapower.to_pandapower` set all generators as
  controllable by default. Generators in converted the pandapower case are named
  using the ``idx`` of ``StaticGen``.
- Bug fixes in ``interop.pandapower.make_link_table()``.

Other changes:

- Added a new service type :py:class:`andes.core.service.SubsService` for
  temporary symbols that will be substituted at code generation time.
- ``TDS.plt.plot()`` now accepts a list of variable objects. For example,
  ``ss.TDS.plt.plot([ss.GENROU.omega, ss.GENROU.delta], a=[0, 1])`` will plot
  the rotor speed and angles of the 0-th and the 1-st generator.
- Added :ref:`REGCP1` model for generic converters with PLL support.
- Fixed PSS/E parser for :ref:`HYGOV`.

v1.6.3 (2022-04-06)
-------------------
- Adjustments in the Pandapower interface. Added ``make_GSF()`` for the
  generation shift factor matrix.
- Reduced import overhead for the command-line tool.

v1.6.2 (2022-03-27)
-------------------
Interoperability:

- Added interoperability modules for MATPOWER (through Oct2Py), pandapower and
  pypowsybl.
- Added Examples and API reference for the interoperability module.
- Improved the setup script to support extra dependencies. The following extras
  groups are supported: ``dev`` and ``interop``. See :ref:`Install_extras` for
  more information.
- Added tests for power flow calculation against MATPOWER.

Others:

- Added a shorthand command ``andes.system.example()`` to return a disposable
  system. It can be useful for quick prototyping.
- Improved the formatting and navigation of Model references.
- Models store the base values for per-unit conversion in ``Model.bases``.

v1.6.1 (2022-03-13)
-------------------
- Revamped documentation with a much improved "Getting started" section.

v1.6.0 (2022-03-11)
-------------------
- Migrated documentation to the pydata template.
- Added compatibility with SymPy 1.9 and 1.10.

v1.5 Notes
==========

v1.5.12 (2022-03-05)
--------------------
- Improved PSS/E parsers for WTDTA1 model to follow PSS/E
  parameter definition.
- Included the Jupyter notebook examples in the documentation.
- Tweaks to the plot utility.

v1.5.11 (2022-02-23)
--------------------
- Reduced the tolerance for tiny variable increments to be
  treated as zero.
- Fixed PSS/E parsers for renewable models.
- Minor renewable model fixes.

v1.5.10 (2022-02-01)
--------------------
- Fixed one equation in `REGC_A`.

v1.5.9 (2022-01-31)
--------------------
- Added PLL1, a simple PLL model.
- Renamed ``REGCVSG`` to ``REGCV1`` and ``REGCVSG2`` to ``REGCV2``.
- Added an alias list for model names. See ``models/__init__.py``.
- Multiprocessing now executes on all CPUs that are physical, instead
  of logical. A new package ``psutil`` needs to be installed.
- Use of ``Selector`` is deprecated.

v1.5.8 (2021-12-21)
--------------------
- Full initialization debug message will be printed only when
  ``-v 10`` and ``run --init`` are both used.
- Improved warning of out-of-limit initialization. Variables
  initialized at limits will be shown only at the debug level.
- Initialization improvements for models ``REGCA1`` and ``REECA1``.
- Added model ``HYGOV``.
- Changed the default ``vout`` of offline exciters to zeros.
  All ``vout`` equations need to be multipled by ``ue``.

v1.5.7 (2021-12-11)
--------------------
This minor release highlights the improved debugging of initialization.

Highly verbose initialization output can be enabled when the verbose
level is 10 or less. For example,

.. code:: bash

    andes -v 10 run test.xlsx -r tds --init

will set the verbose level to 10 and run ``test.xlsx`` in the current
folder, proceed to time-domain simulation but only initialize the models.
Outputs will be printed to the shell where the command is executed.

To save the output to a file, use the following in a UNIX shell:

.. code:: bash

    andes -v 10 run test.xlsx -r tds --init > info.txt 2>&1

where the first ``>`` pipes the output to a file named ``info.txt``,
and ``2>&1`` appends stderr (2) to stdout (1).

The other main improvement is allowing automatic limit adjustment
during initialization. Due to parameter errors, some variables
will be initialized to values outside the given limits.
Most commercial software does not attempt to fix the parameter
but rather adjust the limit in run time.

The same approach is followed in ANDES by automatically adjusting
the upper limit, if exceeded, to variable initial values.
The lower limit, however, is kept unadjusted by default.

Discrete components now take an argument named ``allow_adjust``
so that the model developer can specify if its limits can be
adjusted or must be kept as is.
Each model is allowed to specify three config flags to customize
runtime behaviors: ``allow_adjust``, ``adjust_lower``,
and ``adjust_uppwer``. By default, ``allow_adjust=True``,
``adjust_upper=True``, and ``adjust_lower=False``.
One can modify the config file to enable or disable the
limit adjustments for specific models.

Other fixes include:

- Bug fixes for ``GAST`` parameter ``AT``.
- Bug fixes for ``IEEET3``, ``GAST``, ``ESAC1A`` and ``ESST1A``
  when device is off to avoid matrix singularity.

v1.5.6 (2021-11-25)
--------------------
- Allow specifying config options through command-line arguments
  ``--config-option``.
- Added a voltage and frequency playback model ``PLBVFU1``.
- Bug fixes to an SEXS equation.

v1.5.5 (2021-11-13)
--------------------
- Added a `Timeseries` model for reading timeseries data from xlsx.
- Converted several models into Python packages.
- Bug fixes to TGOV1 equations (#226)

v1.5.4 (2021-11-02)
--------------------
- Fixed a bug in generated ``select`` functions that omitted the
  coefficients of ``__ones``.

v1.5.3 (2021-10-31)
--------------------
- Revered special arguments for the generated ``select`` function.
- Stabilized the argument list of pycode. If the pycode is identical
  to existing ones, the existing file will not be overwritten.
  As a result, compiled code is fully cached.
- Partially seperated time-domain integration method into ``daeint.py``.

v1.5.2 (2021-10-27)
--------------------
- Removed ``CVXOPT`` dependency.
- Removed ``__zeros`` and ``__ones`` as they are no longer needed.

- Added ``andes prep -c`` to precompile the generated code.
- Added utility functions for saving and loading system snapshots.
  See ``andes/utils/snapshot.py``.

- Compiled numba code is always cached.
- Bug fixes.

v1.5.1 (2021-10-23)
--------------------
- Restored compatibility with SymPy 1.6.
- Added a group for voltage compensators.
- New models: ``IEEEVC`` and ``GAST``.

v1.5.0 (2021-10-13)
--------------------
- Support numba just-in-time compilation of all equation and Jacobian calls.

This option accelerates simulations by up to 30%.
The acceleration is visible in medium-scale systems with multiple models.
Such systems involve heavy function calls but a rather moderate load
for linear equation solvers.
The speed up is less significant in large-scale systems where
solving equations is the major time consumer.

Numba is required and can be installed with ``pip install numba`` or
``conda install numba``.

To turn on numba for ANDES, in the ANDES configuration under ``[System]``,
set ``numba = 1`` and ``numba_cache = 1``.

The just-in-time compilation will compile the code upon the first execution
based on the input types.
When compilation is triggered, ANDES may appear frozen due to the compilation lag.
The option ``numba_cache = 1`` will cache compiled machine code, so that
the lag only occurs once until the next ``andes prep``.

- Allow ``BackRef`` to populate to models through ``Group``.

When model `A` stores an ``IdxParam`` pointing to a group, if ``BackRef``
with the name `A` are declared in both the group and the model,
both ``BackRef`` will retrieve the backward references from model `A`.

- Allow ``BaseVar`` to accept partial initializations.

If ``BaseVar.v_str_add = True``, the value of `v_str` will be added in place
to variable value.
An example is that voltage compensator sets part of the input voltage, and
exciter reads the bus voltage. Exciter has `v.v_str_add = True` so that
when compensators exist, the input voltage will be bus voltage (vbus) plus
(Eterm - vbus).
If no compensator exists, exciter will use bus voltages and function as expected.

- Added reserved variable names ``__ones`` and ``__zeros`` for ones and
  zeros with length equal to the device number.

``__ones`` and ``__zeros`` are useful for vectorizing ``choicelist``
in ``Piecewise`` functions.

v1.4 Notes
==========

v1.4.4 (2021-10-05)
--------------------
- Bug fixes for refreshing generated code.

v1.4.3 (2021-09-25)
--------------------
This release features parallel processing that cuts the time for
``andes prepare`` by more than half.

- ``andes prepare`` supports multiprocessing and uses it by default.
- Added aliases ``andes st`` and ``andes prep`` for
  ``andes selftest`` and ``andes prepare``.
- ``andes.config_logger`` supports setting new ``stream_level`` and
  ``file_level``.

New exciter models are contributed by Jinning Wang.

- Added ``AC8B``, ``IEEET3`` and ``ESAC1A``.

Other changes include disallowing numba's ``nopython`` mode.

v1.4.2 (2021-09-12)
--------------------
- Bug fixes
- Dropped support for ``cvxoptklu``.

v1.4.1 (2021-09-12)
--------------------
- Bug fixes.
- Overhaul of the ``prepare`` and ``undill`` methods.
- ``andes prepare`` can be called for specific models through
  ``-m``, which takes one or many model names as arguments.

v1.4.0 (2021-09-08)
--------------------
This release highlights the distributed energy resource protection model.

- Added ``DGPRCT1`` model to provide DG models with voltage-
  and frequency-based protection following IEEE 1547-2018.
- ``REECA1E`` supports frequency droop on power.
- Throws TypeError if type mismatches when using ExtAlgeb and ExtState.

v1.3 Notes
==========

v1.3.12 (2021-08-22)
--------------------
Plot enhancements:

- ``plot()`` takes an argument ``mark`` for masking y-axis data based on
  the ``left`` and ``right`` range parameters.
- ``TDS.plt`` provides a ``panoview`` method for plotting an panoramic view
  for selected variables and devices of a model.

Models:

- Added WIP EV models and protection models.

Test case:
- Added CURENT EI test system.
- Added a number of IEEE 14 bus test systems for specific models.

v1.3.11 (2021-07-27)
--------------------
- Added ``REECA1E`` model with inertia emulation.
- Fixed an issue where the ``vtype`` of services was ignored.
- Changed default DPI for plotting to 100.

v1.3.10 (2021-06-08)
--------------------
- Bug fixes for controllers when generators are off.

v1.3.9 (2021-06-02)
--------------------
- Bug fixes in exciters when generators are offline.
- Added `safe_div` function for initialization equations.

v1.3.8 (2021-06-02)
--------------------
- Added ``REGCVSG`` model for voltage-source controlled renewables.
- Turbine governors are now aware of the generator connection status.

v1.3.7 (2021-05-03)
--------------------
- Allow manually specifying variables needing initialization
  preceding a variable. Specify a list of variable names through
  ``BaseVar.deps``.

v1.3.6 (2021-04-23)
--------------------
- Patched ESD1 model. Converted `distributed.py` into a package.
- Bug fixes.

v1.3.5 (2021-03-20)
--------------------
- Fixed a bug in connectivity check when bus 0 is islanded.
- Updated notebook examples.
- Updated tutorials.

v1.3.4 (2021-03-13)
--------------------
- Fixed a bug for the generated renewable energy code.

v1.3.2 (2021-03-08)
--------------------
- Relaxed the version requirements for NumPy and SymPy.

v1.3.1 (2021-03-07)
--------------------
- Writes all generated Python code to ``~/.andes/pycode`` by default.
- Uses generated Python code by default instead of `calls.pkl`.
- Works with NumPy 1.20; works on Apple Silicon (use `miniforge`) to
  install native Python and NumPy for Apple Silicon.
- Generalized model initialization: automatically determines the
  initialization sequence and solve equations iteratively when
  necessary.
- In `System.config`, `save_pycode` and `use_pycode` are now
  deprecated.


v1.3.0 (2021-02-20)
--------------------
- Allow `State` variable set `check_init=False` to skip
  initialization test. One use case is for integrators
  with non-zero inputs (such as state-of-charge integration).
- Solves power flow for systems with multiple areas, each with
  one Slack generator.
- Added `Jumper` for connecting two buses with zero impedance.
- `REGCA1` and synchronous generators can take power ratio
  parameters `gammap` and `gammaq`.
- New models: `IEESGO` and `IEEET1`, `EXAC4`.
- Refactored exciters, turbine governors, and renewable models
  into modules.


v1.2 Notes
==========

v1.2.9 (2021-01-16)
--------------------
- Added system connectivity check for islanded buses.
- Depend on `openpyxl` for reading excel files since `xlrd` dropped
  support for any format but `xlsx` since v2.0.0.

v1.2.7 (2020-12-08)
--------------------
- Time-domain integration now evaluates anti-windup limiter before
  algebraic residuals. It assures that algebraic residuals are
  calculated with the new state values if pegged at limits.
- Fixed the conditions for Iq ramping in REGC;
  removed ``Iqmax`` and ``Iqmin``.
- Added a new plot function ``plotn`` to allow multiple subplots in
  one figure.
- ``TDS.config.g_scale`` is now now used as a factor for scaling
  algebraic equations for better convergence. Setting it to 1.0
  functions the same as before.

v1.2.6 (2020-12-01)
--------------------
- Added `TGOV1N` model which sums `pref` and `paux` after
  the 1/droop block.
- Added `ZIP` and `FLoad` for dynamic analysis. Need to be initialized
  after power flow.
- Added `DAETimeSeries.get_data()` method.
- Added IEEE 14-bus test cases with solar PV (ieee14_solar.xlsx) and
  Generic Type 3 wind (ieee14_wt3.xlsx).

v1.2.5 (2020-11-19)
--------------------
- Added `Summary` model to allow arbitrary information for
  a test case. Works in `xlsx` and `json` formats.
- PV reactive power limit works. Automatically determines
  the number of PVs to convert if `npv2pq=0`.
- Limiter and AntiWindup limiter can use `sign_upper=-1` and
  `sign_lower=-1` to negate the provided limits.
- Improved error messages for inconsistent data.
- `DAETimeSeries` functions refactored.

v1.2.4 (2020-11-13)
--------------------
- Added switched shunt class `ShuntSw`.
- BaseParam takes `inconvert` and `oconvert` for converting parameter
  elements from and to files.

v1.2.3 (2020-11-02)
--------------------
- Support variable `sys_mva` (system base mva) in equation strings.
- Default support for KVXOPT through ``pip`` installation.

v1.2.2 (2020-11-01)
--------------------
New Models:

- ``PVD1`` model, WECC distributed PV model.
  Supports multiple PVD1 devices on the same bus.
- Added ``ACEc`` model, ACE calculation with continuous freq.

Changes and fixes:

- Renamed `TDS._itm_step` to `TDS.itm_step` as a public API.
- Allow variable `sys_f` (system frequency) in equation strings.
- Fixed ACE equation.
  measurement.
- Support ``kvxopt`` as a drop-in replacement for ``cvxopt``
  to bring KLU to Windows (and other platforms).
- Added ``kvxopt`` as a dependency for PyPI installation.

v1.2.1 (2020-10-11)
--------------------
- Renamed `models.non_jit` to `models.file_classes`.
- Removed `models/jit.py` as models have to be loaded and instantiated
  anyway before undill.
- Skip generating empty equation calls.

v1.2.0 (2020-10-10)
--------------------
This version contains major refactor for speed improvement.

- Refactored Jacobian calls generation so that for each model, one call
  is generated for each Jacobian type.
- Refactored Service equation generation so that the exact arguments are
  passed.

Also contains an experimental Python code dump function.

- Controlled in ``System.config``, one can turn on ``save_pycode`` to dump
  equation and Jacobian calls to ``~/.andes/pycode``. Requires one call to
  ``andes prepare``.
- The Python code dump can be reformatted with ``yapf`` through the config
  option ``yapf_pycode``. Requires separate installation.
- The dumped Python code can be used for subsequent simulations through
  the config option ``use_pycode``.

v1.1 Notes
==========

v1.1.5 (2020-10-08)
--------------------
- Allow plotting to existing axes with the same plot API.
- Added TGOV1DB model (TGOV1 with an input dead-band).
- Added an experimental numba support.
- Patched `LazyImport` for a snappier command-line interface.
- ``andes selftest -q`` now skips code generation.

v1.1.4 (2020-09-22)
--------------------
- Support `BackRef` for groups.
- Added CLI ``--pool`` to use ``multiprocess.Pool`` for multiple cases.
  When combined with ``--shell``, ``--pool`` returns ``System`` Objects
  in the list ``system``.
- Fixed bugs and improved manual.

v1.1.3 (2020-09-05)
--------------------
- Improved documentation.
- Minor bug fixes.

v1.1.2 (2020-09-03)
--------------------
- Patched time-domain for continuing simulation.

v1.1.1 (2020-09-02)
--------------------
- Added back quasi-real-time speed control through `--qrt`
  and `--kqrt KQRT`.
- Patched the time-domain routine for the final step.

v1.1.0 (2020-09-01)
--------------------
- Defaulted `BaseVar.diag_eps` to `System.Config.diag_eps`.
- Added option `TDS.config.g_scale` to allow for scaling the
  algebraic mismatch with step size.
- Added induction motor models `Motor3` and `Motor5` (PSAT models).
- Allow a PFlow-TDS model to skip TDS initialization by setting
  `ModelFlags.tds_init` to False.
- Added Motor models `Motor3` and `Motor5`.
- Imported `get_case` and `list_cases` to the root package level.
- Added test cases (Kundur's system) with wind.

Added Generic Type 3 wind turbine component models:

- Drive-train models `WTDTA1` (dual-mass model) and `WTDS`
  (single-mass model).
- Aerodynamic model `WTARA1`.
- Pitch controller model `WTPTA1`.
- Torque (a.k.a. Pref) model `WTTQA1`.


v1.0 Notes
==========

v1.0.8 (2020-07-29)
--------------------
New features and models:

- Added renewable energy models `REECA1` and `REPCA1`.
- Added service `EventFlag` which automatically calls events
  if its input changes.
- Added service `ExtendedEvent` which flags an extended event
  for a given time.
- Added service `ApplyFunc` to apply a numeric function.
  For the most cases where one would need `ApplyFunc`,
  consider using `ConstService` first.
- Allow `selftest -q` for quick selftest by skipping codegen.
- Improved time stepping logic and convergence tests.
- Updated examples.

Default behavior changes include:

- ``andes prepare`` now takes three mutually exclusive arguments,
  `full`, `quick` and `incremental`. The command-line now defaults
  to the quick mode. ``andes.prepare()`` still uses the full mode.
- ``Model.s_update`` now evaluates the generated and the
  user-provided calls in sequence for each service in order.
- Renamed model `REGCAU1` to `REGCA1`.

v1.0.7 (2020-07-18)
--------------------
- Use in-place assignment when updating Jacobian values in Triplets.
- Patched a major but simple bug where the Jacobian refactorization
  flag is set to the wrong place.
- New models: PMU, REGCAU1 (tests pending).
- New blocks: DeadBand1, PIFreeze, PITrackAW, PITrackAWFreeze (tests
  pending), and LagFreeze (tests pending).
- `andes plot` supports dashed horizontal and vertical lines through
  `hline1`, `hline2`, `vline1` and `vline2`.
- Discrete: renamed `DeadBand` to `DeadBandRT` (deadband with
  return).
- Service: renamed `FlagNotNone` to `FlagValue` with an option
  to flip the flags.
- Other tweaks.

v1.0.6 (2020-07-08)
--------------------
- Patched step size adjustment algorithm.
- Added Area Control Error (ACE) model.

v1.0.5 (2020-07-02)
--------------------
- Minor bug fixes for service initialization.
- Added a wrapper to call TDS.fg_update to
  allow passing variables from caller.
- Added pre-event time to the switch_times.

v1.0.4 (2020-06-26)
--------------------
- Implemented compressed NumPy format (npz) for time-domain
  simulation output data file.
- Implemented optional attribute `vtype` for specifying data type
  for Service.
- Patched COI speed initialization.
- Patched PSS/E parser for two-winding transformer winding and
  impedance modes.

v1.0.3 (2020-06-02)
--------------------
- Patches `PQ` model equations where the "or" logic "|" is ignored in
  equation strings. To adjust PQ load in time domain simulation, refer
  to the note in `pq.py`.
- Allow `Model.alter` to update service values.

v1.0.2 (2020-06-01)
--------------------
- Patches the conda-forge script to use SymPy < 1.6. After SymPy version
  1.5.1, comparison operations cannot be sympified. Pip installations are
  not affected.

v1.0.1 (2020-05-27)
--------------------
- Generate one lambda function for each of f and g, instead of generating
  one for each single f/g equation. Requires to run `andes prepare` after
  updating.

v1.0.0 (2020-05-25)
--------------------
This release is going to be tagged as v0.9.5 and later tagged as v1.0.0.

- Added verification results using IEEE 14-bus, NPCC, and WECC systems
  under folder `examples`.
- Patches GENROU and EXDC2 models.
- Updated test cases for WECC, NPCC IEEE 14-bus.
- Documentation improvements.
- Various tweaks.

Pre-v1.0.0
==========

v0.9.4 (2020-05-20)
-------------------

- Added exciter models EXST1, ESST3A, ESDC2A, SEXS, and IEEEX1,
  turbine governor model IEEEG1 (dual-machine support), and stabilizer
  model ST2CUT.
- Added blocks HVGate and LVGate with a work-around for sympy.maximum/
  minimum.
- Added services `PostInitService` (for storing initialized values), and
  `VarService` (variable services that get updated) after limiters and before
  equations).
- Added service `InitChecker` for checking initialization values against
  typical values. Warnings will be issued when out of bound or equality/
  inequality conditions are not met.
- Allow internal variables to be associated with a discrete component which
  will be updated before initialization (through `BaseVar.discrete`).
- Allow turbine governors to specify an optional `Tn` (turbine rating). If
  not provided, turbine rating will fall back to `Sn` (generator rating).
- Renamed `OptionalSelect` to `DataSelect`; Added `NumSelect`, the array-based
  version of `DataSelect`.
- Allow to regenerate code for updated models through ``andes prepare -qi``.
- Various patches to allow zeroing out time constants in transfer functions.

v0.9.3 (2020-05-05)
--------------------
This version contains bug fixes and performance tweaks.

- Fixed an `AntiWindup` issue that causes variables to stuck at limits.
- Allow ``TDS.run()`` to resume from a stopped simulation and run to the new
  end time in ``TDS.config.tf``.
- Improved TDS data dump speed by not constructing DataFrame by default.
- Added tests for `kundur_full.xlsx` and `kundur_aw.xlsx` to ensure
  results are the same as known values.
- Other bug fixes.

v0.9.1 (2020-05-02)
--------------------
This version accelerates computations by about 35%.

- Models with flag ``collate=False``, which is the new default,
  will slice DAE arrays for all internal vars to reduce copying back and forth.
- The change above greatly reduced computation time.
  For ``kundur_ieeest.xlsx``, simulation time is down from 2.50 sec to 1.64 sec.
- The side-effects include a change in variable ordering in output lst file.
  It also eliminated the feasibility of evaluating model equations in
  parallel, which has not been implemented and does not seem promising in Python.
- Separated symbolic processor and documentation generator from Model into
  ``SymProcessor`` and ``Documenter`` classes.
- ``andes prepare`` now shows progress in the console.
- Store exit code in ``System.exit_code`` and returns to system when called
  from CLI.
- Refactored the solver interface.
- Patched Config.check for routines.
- SciPy Newton-Krylov power flow solver is no longer supported.
- Patched a bug in v0.9.0 related to `dae.Tf`.

v0.8.8 (2020-04-28)
--------------------
This update contains a quick but significant fix to boost the simulation speed by avoiding
calls to empty user-defined numerical calls.

- In `Model.flags` and `Block.flags`, added `f_num`, `g_num` and `j_num` to indicate
  if user-defined numerical calls exist.
- In `Model.f_update`, `Model.g_update` and `Model.j_update`, check the above flags
  to avoid unnecessary calls to empty numeric functions.
- For the `kundur_ieeest.xlsx` case, simulation time was reduced from 3.5s to 2.7s.

v0.8.7 (2020-04-28)
--------------------
- Changed `RefParam` to a service type called `BackRef`.
- Added `DeviceFinder`, a service type to find device idx when not provided.
  `DeviceFinder` will also automatically add devices if not found.
- Added `OptionalSelect`, a service type to select optional parameters if provided
  and select fallback ones otherwise.
- Added discrete types `Derivative`, `Delay`, and `Average`,
- Implemented full IEEEST stabilizer.
- Implemented COI for generator speed and angle measurement.

v0.8.6 (2020-04-21)
--------------------
This release contains important documentation fixes and two new blocks.

- Fixed documentations in `andes doc` to address a misplacement of symbols and equations.
- Converted all blocks to the division-free formulation (with `dae.zf` renamed to `dae.Tf`).
- Fixed equation errors in the block documentation.
- Implemented two new blocks: Lag2ndOrd and LeadLag2ndOrd.
- Added a prototype for IEEEST stabilizer with some fixes needed.

v0.8.5 (2020-04-17)
--------------------
- Converted the differential equations to the form of ``T \dot{x} = f(x, y)``, where T is supplied to
  ``t_const`` of ``State/ExtState``.
- Added the support for Config fields in documentation (in ``andes doc`` and on readthedocs).
- Added Config consistency checking.
- Converted `Model.idx` from a list to `DataParam`.
- Renamed the API of routines (summary, init, run, report).
- Automatically generated indices now start at 1 (i.e., "GENCLS_1" is the first GENCLS device).
- Added test cases for WECC system. The model with classical generators is verified against TSAT.
- Minor features: `andes -v 1` for debug output with levels and line numbers.

v0.8.4 (2020-04-07)
--------------------
- Added support for JSON case files. Convert existing case file to JSON with ``--convert json``.
- Added support for PSS/E dyr files, loadable with ``-addfile ADDFILE``.
- Added ``andes plot --xargs`` for searching variable name and plotting. See example 6.
- Various bug fixes: Fault power injection fix;

v0.8.3 (2020-03-25)
--------------------
- Improved storage for Jacobian triplets (see ``andes.core.triplet.JacTriplet``).
- On-the-fly parameter alteration for power flow calculations (``Model.alter`` method).
- Exported frequently used functions to the root package
  (``andes.config_logger``, ``andes.run``, ``andes.prepare`` and ``andes.load``).
- Return a list of System objects when multiprocessing in an interactive environment.
- Exported classes to `andes.core`.
- Various bug fixes and documentation improvements.

v0.8.0 (2020-02-12)
--------------------
- First release of the hybrid symbolic-numeric framework in ANDES.
- A new framework is used to describe DAE models, generate equation documentation, and generate code for
  numerical simulation.
- Models are written in the new framework. Supported models include GENCLS, GENROU, EXDC2, TGOV1, TG2
- PSS/E raw parser, MATPOWER parser, and ANDES xlsx parser.
- Newton-Raphson power flow, trapezoidal rule for numerical integration, and full eigenvalue analysis.

v0.6.9 (2020-02-12)
--------------------

- Version 0.6.9 is the last version for the numeric-only modeling framework.
- This version will not be updated any more.
  But, models, routines and functions will be ported to the new version.