## Modular Refactorize
*   Varout to DataFrame
*   Optimize `_varname()` in models


## Performance Improvement
*   Reduce the overhead in `VarOut.store()` and `TDS.dump_results()`
    *   Impact: Low

*   Reduce the overhead in `DAE.reset_Ac()`
    *   Impact: High


## New Functions
*   Root loci plots
*   Eigenvalue analysis report sort options: by damping, frequency, or eigenvalue


## Version 0.7.0

### Milestones
- [x] A working `PQNew` class with an option to convert to Z; Allow config in models
- [x] A defined data loading, model initialization, variable/equation relaying sequence
- [x] A working power flow routine fully generated from symbolic expressions
- [x] A working `System` class providing parameter retrieval by group and model
- [x] Time domain simulation using scipy.integrate (odeint and solve_ivp)

### To-do bullets
- [x] Clearly define interface variables `VarExt`
- [x] Define associated equation with each variable (internal of interface)
- [x] Use SymPy/SynEngine to generate function calls - define the interfaces
- [x] Use SymEngine to get the derivative for each model; the derivatives may store in a smaller matrix locally to the model
- [x] Pickle/dill the auto-generated function calls on the first run
- [x] Function for providing the jacobian triplet
- [x] Implement a PV model with the old code - Partially done with the Hybrid `j_numeric`
- [x] Define the call sequence for data flow between models and dae/models
- [x] Initial values, pass initialized values between models, and solve initializer equations
- [x] Improve the access to dae attributes. Get rid of `self.dae.__dict__[f'']` usages.
- [x] Configs in models that can participate in the computation, saved to files, and loaded
- [x] Dummy service variables for storing initial values (such as pm0, vref0) (Solved with `Service` itself)
- [x] Improve running functions based on routine. Get rid of passing around `is_tds`. Possibly implementing a
 `pflow_models` and a `tds_models` list in system
- [x] Implement a group for PV and Slack. Possibly implementing all groups in `group.py`
- [x]   Implement index handling in `group.py`, update `link_external` to work with groups (Let group implement
 similar api to models.idx2uid, or implement a `get_by_idx` for both)
- [x] Let Synchronous generator subsitute PV; more generally, let any non-pflow model substitute a pflow one
 (Currently done with `v_numeric`)
- [x] Prototype Connectivity checking, previously implemented as `gyisland`. Need to be implemented under
 `System` (It turns out that `gy_island` and `gisland` does not need to be implemented. If there is a large
  mismatch, just let the power flow fail and let the user correct the data.)
- [x] Prototype a Bus voltage average for area (COI-type of one-to-multiple aggregation model 
(RefParam, SericeReduce and ServiceRepeat)
- [x] Divide limiter to update_var and update_eq (such as anti-windup limiter which depends on equations)
- [x] Allow time definition in models reserve keyword `dae_t` (see `Area`)
*   Clean up `System._get_models`; Clean up the use of model-dependent calls
*   Clearly define `reset` and `clear`
*   Clean up the use of `vars_to_dae` and `vars_to_models` everywhere. 
*   Sequantial initialization - Reduce the data back and forth in `System.initialize()`. Maybe delay
 `link_external` for variables until they are initialized
*   Fix the case when Area is not a `TDS` model but has an equation `time = dae_t` which does not get updated
 during TDS.
*   Implement a trapezoidal rule for numerical integration
*   Allow for semi-implicit method formulation
*   Allow for semi-analytical derivation of equations
*   Define general hooks - when should the connectivity check happen
*   Add a more generic parser for PSSE RAW
*   Store states in systems to enforce sequence of execution

### Examples
- [x] implement a standalone PI controller with numerical jacobians

### Features
- [x] Newton-Raphson and Newton-Krylov power flow (with handling of separated systems)

## Version 0.7.1
## Milestones

### To-do bullets
*   A working `GENROU` model with saturation function
*   A refreshed raw file reader to build data into `ModelData`
*   A refreshed dyr file reader

## Later Versions
*   Solve non-linear initialization equations
*   Find a workaround for IDA