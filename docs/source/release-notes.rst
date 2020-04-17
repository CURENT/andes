=============
Release Notes
=============

The APIs before v1.0.0 are in beta and may change without prior notice.

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