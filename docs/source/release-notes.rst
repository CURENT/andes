=============
Release Notes
=============

The APIs before v3.0.0 are in beta and may change without prior notice.

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
-------------------
This version contains bug fixes and performance tweaks.

- Fixed an `AntiWindup` issue that causes variables to stuck at limits.
- Allow ``TDS.run()`` to resume from a stopped simulation and run to the new
  end time in ``TDS.config.tf``.
- Improved TDS data dump speed by not constructing DataFrame by default.
- Added tests for `kundur_full.xlsx` and `kundur_aw.xlsx` to ensure
  results are the same as known values.
- Other bug fixes.

v0.9.1 (2020-05-02)
-------------------
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
-------------------
This update contains a quick but significant fix to boost the simulation speed by avoiding
calls to empty user-defined numerical calls.

- In `Model.flags` and `Block.flags`, added `f_num`, `g_num` and `j_num` to indicate
  if user-defined numerical calls exist.
- In `Model.f_update`, `Model.g_update` and `Model.j_update`, check the above flags
  to avoid unnecessary calls to empty numeric functions.
- For the `kundur_ieeest.xlsx` case, simulation time was reduced from 3.5s to 2.7s.

v0.8.7 (2020-04-28)
-------------------
- Changed `RefParam` to a service type called `BackRef`.
- Added `DeviceFinder`, a service type to find device idx when not provided.
  `DeviceFinder` will also automatically add devices if not found.
- Added `OptionalSelect`, a service type to select optional parameters if provided
  and select fallback ones otherwise.
- Added discrete types `Derivative`, `Delay`, and `Average`,
- Implemented full IEEEST stabilizer.
- Implemented COI for generator speed and angle measurement.

v0.8.6 (2020-04-21)
-------------------
This release contains important documentation fixes and two new blocks.

- Fixed documentations in `andes doc` to address a misplacement of symbols and equations.
- Converted all blocks to the division-free formulation (with `dae.zf` renamed to `dae.Tf`).
- Fixed equation errors in the block documentation.
- Implemented two new blocks: Lag2ndOrd and LeadLag2ndOrd.
- Added a prototype for IEEEST stabilizer with some fixes needed.

v0.8.5 (2020-04-17)
-------------------
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
-------------------
- Added support for JSON case files. Convert existing case file to JSON with ``--convert json``.
- Added support for PSS/E dyr files, loadable with ``-addfile ADDFILE``.
- Added ``andes plot --xargs`` for searching variable name and plotting. See example 6.
- Various bug fixes: Fault power injection fix;

v0.8.3 (2020-03-25)
-------------------
- Improved storage for Jacobian triplets (see ``andes.core.triplet.JacTriplet``).
- On-the-fly parameter alteration for power flow calculations (``Model.alter`` method).
- Exported frequently used functions to the root package
  (``andes.config_logger``, ``andes.run``, ``andes.prepare`` and ``andes.load``).
- Return a list of System objects when multiprocessing in an interactive environment.
- Exported classes to `andes.core`.
- Various bug fixes and documentation improvements.

v0.8.0 (2020-02-12)
-------------------
- First release of the hybrid symbolic-numeric framework in ANDES.
- A new framework is used to describe DAE models, generate equation documentation, and generate code for
  numerical simulation.
- Models are written in the new framework. Supported models include GENCLS, GENROU, EXDC2, TGOV1, TG2
- PSS/E raw parser, MATPOWER parser, and ANDES xlsx parser.
- Newton-Raphson power flow, trapezoidal rule for numerical integration, and full eigenvalue analysis.

v0.6.9 (2020-02-12)
-------------------
- Version 0.6.9 is the last version for the numeric-only modeling framework.
- This version will not be updated any more.
  But, models, routines and functions will be ported to the new version.