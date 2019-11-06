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
*   A working `PQNew` class with an option to convert to Z
*   A working `System` class providing parameter retrieval by group and model
*   A refreshed raw file reader to build data into `ModelData`
*   A defined data loading, model initialization, variable/equation relaying sequence
*   A working power flow routine fully generated from symbolic expressions
*   A working `GENROU` model with saturation function


### To-do bullets
- [x] Clearly define interface variables `VarExt`
- [x] Define associated equation with each variable (internal of interface)
- [x] Use SymPy/SynEngine to generate function calls - define the interfaces
- [x] Use SymEngine to get the derivative for each model; the derivatives may store in a smaller matrix locally to the model
- [x] Pickle/dill the auto-generated function calls on the first run
- [x] Function for providing the jacobian triplet
*   Implement a PV model with the old code
*   Implement a group for PV and Slack
*   Define the call sequence for data flow between models and dae/models
*   Prototype Connectivity checking, previously implemented as `gyisland`
*   Initial values, pass initialized values between models, and solve initializer equations
*   Allow semi-implicit method formulation
*   Allow semi-analytic derivation of equations

### Examples
- [x] implement a standalone PI controller with numerical jacobians

Version 0.7.1
## Milestones

### To-do bullets
*  A refreshed dyr file reader