.. _ReleaseNotes:

=============
Release Notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

v1.2 Notes
----------

v1.2.0 (2020-10-10)
```````````````````
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
----------
v1.1.5 (2020-10-08)
```````````````````
- Allow plotting to existing axes with the same plot API.
- Added TGOV1DB model (TGOV1 with an input dead-band).
- Added an experimental numba support.
- Patched `LazyImport` for a snappier command-line interface.
- ``andes selftest -q`` now skips code generation.

v1.1.4 (2020-09-22)
```````````````````
- Support `BackRef` for groups.
- Added CLI ``--pool`` to use ``multiprocess.Pool`` for multiple cases.
  When combined with ``--shell``, ``--pool`` returns ``System`` Objects
  in the list ``system``.
- Fixed bugs and improved manual.

v1.1.3 (2020-09-05)
```````````````````
- Improved documentation.
- Minor bug fixes.

v1.1.2 (2020-09-03)
```````````````````
- Patched time-domain for continuing simulation.

v1.1.1 (2020-09-02)
```````````````````
- Added back quasi-real-time speed control through `--qrt`
  and `--kqrt KQRT`.
- Patched the time-domain routine for the final step.

v1.1.0 (2020-09-01)
```````````````````
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
----------

v1.0.8 (2020-07-29)
```````````````````
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
```````````````````
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
```````````````````
- Patched step size adjustment algorithm.
- Added Area Control Error (ACE) model.

v1.0.5 (2020-07-02)
```````````````````
- Minor bug fixes for service initialization.
- Added a wrapper to call TDS.fg_update to
  allow passing variables from caller.
- Added pre-event time to the switch_times.

v1.0.4 (2020-06-26)
```````````````````
- Implemented compressed NumPy format (npz) for time-domain
  simulation output data file.
- Implemented optional attribute `vtype` for specifying data type
  for Service.
- Patched COI speed initialization.
- Patched PSS/E parser for two-winding transformer winding and
  impedance modes.

v1.0.3 (2020-06-02)
```````````````````
- Patches `PQ` model equations where the "or" logic "|" is ignored in
  equation strings. To adjust PQ load in time domain simulation, refer
  to the note in `pq.py`.
- Allow `Model.alter` to update service values.

v1.0.2 (2020-06-01)
```````````````````
- Patches the conda-forge script to use SymPy < 1.6. After SymPy version
  1.5.1, comparison operations cannot be sympified. Pip installations are
  not affected.

v1.0.1 (2020-05-27)
```````````````````
- Generate one lambda function for each of f and g, instead of generating
  one for each single f/g equation. Requires to run `andes prepare` after
  updating.

v1.0.0 (2020-05-25)
```````````````````
This release is going to be tagged as v0.9.5 and later tagged as v1.0.0.

- Added verification results using IEEE 14-bus, NPCC, and WECC systems
  under folder `examples`.
- Patches GENROU and EXDC2 models.
- Updated test cases for WECC, NPCC IEEE 14-bus.
- Documentation improvements.
- Various tweaks.

Pre-v1.0.0
----------

v0.9.4 (2020-05-20)
```````````````````

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
```````````````````
This version contains bug fixes and performance tweaks.

- Fixed an `AntiWindup` issue that causes variables to stuck at limits.
- Allow ``TDS.run()`` to resume from a stopped simulation and run to the new
  end time in ``TDS.config.tf``.
- Improved TDS data dump speed by not constructing DataFrame by default.
- Added tests for `kundur_full.xlsx` and `kundur_aw.xlsx` to ensure
  results are the same as known values.
- Other bug fixes.

v0.9.1 (2020-05-02)
```````````````````
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
```````````````````
This update contains a quick but significant fix to boost the simulation speed by avoiding
calls to empty user-defined numerical calls.

- In `Model.flags` and `Block.flags`, added `f_num`, `g_num` and `j_num` to indicate
  if user-defined numerical calls exist.
- In `Model.f_update`, `Model.g_update` and `Model.j_update`, check the above flags
  to avoid unnecessary calls to empty numeric functions.
- For the `kundur_ieeest.xlsx` case, simulation time was reduced from 3.5s to 2.7s.

v0.8.7 (2020-04-28)
```````````````````
- Changed `RefParam` to a service type called `BackRef`.
- Added `DeviceFinder`, a service type to find device idx when not provided.
  `DeviceFinder` will also automatically add devices if not found.
- Added `OptionalSelect`, a service type to select optional parameters if provided
  and select fallback ones otherwise.
- Added discrete types `Derivative`, `Delay`, and `Average`,
- Implemented full IEEEST stabilizer.
- Implemented COI for generator speed and angle measurement.

v0.8.6 (2020-04-21)
```````````````````
This release contains important documentation fixes and two new blocks.

- Fixed documentations in `andes doc` to address a misplacement of symbols and equations.
- Converted all blocks to the division-free formulation (with `dae.zf` renamed to `dae.Tf`).
- Fixed equation errors in the block documentation.
- Implemented two new blocks: Lag2ndOrd and LeadLag2ndOrd.
- Added a prototype for IEEEST stabilizer with some fixes needed.

v0.8.5 (2020-04-17)
```````````````````
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
```````````````````
- Added support for JSON case files. Convert existing case file to JSON with ``--convert json``.
- Added support for PSS/E dyr files, loadable with ``-addfile ADDFILE``.
- Added ``andes plot --xargs`` for searching variable name and plotting. See example 6.
- Various bug fixes: Fault power injection fix;

v0.8.3 (2020-03-25)
```````````````````
- Improved storage for Jacobian triplets (see ``andes.core.triplet.JacTriplet``).
- On-the-fly parameter alteration for power flow calculations (``Model.alter`` method).
- Exported frequently used functions to the root package
  (``andes.config_logger``, ``andes.run``, ``andes.prepare`` and ``andes.load``).
- Return a list of System objects when multiprocessing in an interactive environment.
- Exported classes to `andes.core`.
- Various bug fixes and documentation improvements.

v0.8.0 (2020-02-12)
```````````````````
- First release of the hybrid symbolic-numeric framework in ANDES.
- A new framework is used to describe DAE models, generate equation documentation, and generate code for
  numerical simulation.
- Models are written in the new framework. Supported models include GENCLS, GENROU, EXDC2, TGOV1, TG2
- PSS/E raw parser, MATPOWER parser, and ANDES xlsx parser.
- Newton-Raphson power flow, trapezoidal rule for numerical integration, and full eigenvalue analysis.

v0.6.9 (2020-02-12)
```````````````````
- Version 0.6.9 is the last version for the numeric-only modeling framework.
- This version will not be updated any more.
  But, models, routines and functions will be ported to the new version.