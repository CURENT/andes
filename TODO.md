# TODO

## Future Enhancements

The following items represent potential enhancements for future versions
of ANDES. Contributions are welcome.

* Time-series data input support (e.g., a `PQTS` model for loads with
  time-varying profiles)
* Eigenvalue analysis report with sort options (by damping ratio,
  frequency, or eigenvalue magnitude)
* Automatic generation of block diagrams from symbolic model definitions

## Completed

The items below have been completed in previous releases and are retained
for reference.

* [X] Center-of-inertia (COI) model for rotor angle and speed
* [X] Selection of output variables via the `Output` model
* [X] Root loci plots in eigenvalue analysis
* [X] Renewable generator models (REGCA1, REECA1, REPCA1)
* [X] Switched shunt model (ShuntSw)
* [X] Numba JIT compilation for all models (`andes prep -c`)
* [X] Symbolic DAE modeling with automated code generation
* [X] Block library with common transfer functions
* [X] Discrete component library (limiters, deadbands, anti-windup)
* [X] Newton-Raphson and Newton-Krylov power flow
* [X] Trapezoidal method for time-domain simulation
* [X] Per-unit conversion and sequential initialization
* [X] PSS/E RAW and DYR file parsing
