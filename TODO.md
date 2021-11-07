# v1.5

## Milestones

* Separate array-enabled parameters from NumParam into
  `BaseConvParam`, which implements `iconvert` and `oconvert`,
  and then `ListParam`
[x] Improve the robustness of accessing the fields of `pycode` by properly
    handling `KeyError`s.
[x] Numba code generation for all models. Add a function for triggering
    all the JIT compilation before simulation. `andes prep -c` does it.
* Unit Model Test framework - a difficult task. Testing models will
  require proper inputs (from external models), addressing from the system,
  proper initialization, and the coupling with integration methods.

# Later Versions

* Center-of-inertia for rotor angle and speed.
* Allow selection of output variables
* Timeseries data input support. Reading data from file.
* `PQTS` model for PQ with time-series.
* Generalize two-terminal and multi-terminal devices
* Allow for semi-analytical derivation of equations
* Define general hooks - when should the connectivity check happen
* Vectorize the `enable` option for most discrete blocks.
* allow to remind developer of missing equations (such as `vout` of Exciters).

## Help Wanted

* Restore compatibility with dome format
* Draw block diagram from symbolic models using BDP (or SchemDraw)
* Eigenvalue analysis report sort options: by damping, frequency, or eigenvalue
* Root loci plots

# Previous

## v1.4

* [X] Disallow linking ExtAlgeb to State and ExtState to Algeb (check in prepare).

## v1.3

## Milestones

* [X] Implement switched shunt model (ShuntSw)
* [X] Allow arrays as device parameter
* [X] An `Summary` model for storing system summary
* [X] Allow export of equation values. Using equation values are not recommended
  due to convergence considerations (partial derivatives not used).
* [X] Robust iterative initialization for variables

## Version 1.1.0 (completed)

## Milestones

* [X] Limiter check and warning after initialization
* [X] Renewable generator models (REGC_A, REEC_A, REPC_A)

## To-do list

* [X] Improve the speed of `DAETimeSeries.unpack`.
* [X] Allow simulating to a time, pause, and continue to a new ending time.
* [X] Store TDS data as NumPy compressed format `npz`; allow to reload both `npy` and `npz`.
* [X] Power flow model variables erroneously point to the old `dae.y` array (fixed)
* [X] Allow loading default config for `selftest` (andes.run(default_config=True))

## Version 0.9.0 (completed)

### Milestones

* [X] Help system and consistency check system for Config
* [X] Handling of zero time constants (through `State.t_const`)
* [X] Refactor Model to separate symbolic processing part as `ModelSymbolic` -> `Model.syms`.
* [X] Separate the solver class into an interface class + different solver classes

### To-do bullets

* [X] A working `GENROU` model with saturation function
* [X] Fix the model connectivity status `u` in interface equations
* [X] A refreshed raw file reader to build data into `ModelData` (partially refreshed)
* [X] A refreshed dyr file reader
* [X] Add ``Model.int`` for internal indexer (implemented by replacing `Model.idx` as DataParam)
* [X] Allow adding routine without modifying code (as long as routines are added to `Routines.all_routines`)
* [X] Add a help system for Config
* [X] Add consistency checks for Config
* [X] Return an error state to system if a simulation routine fails
* [X] Example COI model

## Version 0.8.0 (Completed)

### Milestones

* [X] A working `PQNew` class with an option to convert to Z; Allow config in models
* [X] A defined data loading, model initialization, variable/equation relaying sequence
* [X] A working power flow routine fully generated from symbolic expressions
* [X] A working `System` class providing parameter retrieval by group and model
* [X] Time domain simulation using scipy.integrate (odeint and solve_ivp)

### Features

* [X] Symbolic DAE modeling and automated code generation for numerical simulation
* [X] Numerical DAE modeling for scenarios when symbolic implementations are difficult
* [X] Rapid modeling with block library with common transfer functions.
* [X] Discrete component library such as hard limiter, dead band, and anti-windup limiter.
* [X] Pretty printing of DAE and automatically derived Jacobians
* [X] Newton-Raphson and Newton-Krylov power flow (with automatic handling of separated systems).
* [X] Trapezoidal method for semi-explicit time domain simulation.

### Usability

* [X] Set up command line interface

### To-do bullets

* [X] Clearly define interface variables `VarExt`
* [X] Define associated equation with each variable (internal of interface)
* [X] Use SymPy/SynEngine to generate function calls - define the interfaces
* [X] Use SymEngine to get the derivative for each model; the derivatives may store in a smaller matrix locally to the model
* [X] Pickle/dill the auto-generated function calls on the first run
* [X] Function for providing the jacobian triplet
* [X] Implement a PV model with the old code - Partially done with the Hybrid `j_numeric`
* [X] Define the call sequence for data flow between models and dae/models
* [X] Initial values, pass initialized values between models, and solve initializer equations
* [X] Improve the access to dae attributes. Get rid of `self.dae.__dict__[f'']` usages.
* [X] Configs in models that can participate in the computation, saved to files, and loaded
* [X] Dummy service variables for storing initial values (such as pm0, vref0) (Solved with `Service` itself)
* [X] Improve running functions based on routine. Get rid of passing around `is_tds`. Possibly implementing a

`pflow_models` and a `tds_models` list in system

* [X] Implement a group for PV and Slack. Possibly implementing all groups in `group.py`
* [X] Implement index handling in `group.py`, update `link_external` to work with groups (Let group implement

similar api to models.idx2uid, or implement a `get_by_idx` for both)

* [X] Let Synchronous generator subsitute PV; more generally, let any non-pflow model substitute a pflow one

(Currently done with `v_numeric` )

* [X] Prototype Connectivity checking, previously implemented as `gyisland`. Need to be implemented under

`System` (It turns out that `gy_island` and `gisland` does not need to be implemented. If there is a large
mismatch, just let the power flow fail and let the user correct the data.)

* [X] Prototype a Bus voltage average for area (COI-type of one-to-multiple aggregation model

(RefParam, SericeReduce and RepeaterService)

* [X] Divide limiter to update_var and update_eq (such as anti-windup limiter which depends on equations)
* [X] Allow time definition in models reserve keyword `dae_t` (see `Area`)
* [X] Clean up `System._get_models`; Clean up the use of model-dependent calls
* [X] Clearly define `reset` and `clear` (`clear` clears the values but retains the size;  `reset` resets

attributes to a state before setup)

* [X] Fix the case when Area is not a `TDS` model but has an equation `time = dae_t` which does not get updated

during TDS. (a non- `tds` model is not allowed to use `dae.t` )

* [X] Implement a trapezoidal rule for numerical integration
* [X] Refactorize jacobian after critical time (works for implicit method)
* [X] Use an iterator into `System.times` rather than deleting `times[0]`
* [X] Implement a time-based switching model and export all switching time to `System.switch_times`
* [X] Sequential initialization
* [X] Limiter in PI controller
* [X] Clean up the use of `vars_to_dae` and `vars_to_model`
* [X] low pass filter in PI Controller - How the equations should be written
* [X] Refactor `Config` to make load and save part of the config
* [X] Per-unit conversion (get ExtParam Sn, Vn before per unit conversion - Yes)
* [X] LaTeX names in blocks and limiters
* [X] Input switch. If input equals 1, 2, 3, or 4; Discrete
* [X] Quadratic and exponential saturation for generators
* [X] Piecewise nonlinear functions (block. Piecewise)
* [X] Decide whether to keep Calc as Var or move it to service. (eliminated from VarBase subclasses; Likely a

service subclass)

* [X] Use SymPy to solve e1d, e1q, e2d and e2q equations for GENROU
* [X] Test initialization and report suspect issues
* [X] Test anti-windup limiter
* [X] Added expression symbol checking. Undefined symbols will throw ValueError during preparation
* [X]`System.reset()` not working after `TDS.run`
* [X] Export power flow iteration steps for debugging; export limiter status (get_inputs) alongside equations (implemented in _input_z)
* [X] Batch simulation with in-place parameter modification (implemented with `Model.alter()`)
* [X] Control feedback, possibly with perturbation files (control implemented in this approach has a "delay" of a step size)
* [X] TimeSeries output to DataFrame (system.dae.ts.df)

### Examples

* [X] implement a standalone PI controller with numerical jacobians
