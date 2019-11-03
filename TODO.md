## Modular Refactorization

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
*   Clearly define interface variables`VarExt`
*   Define associated equation with each variable (internal of interface)
*   Define the call sequence for data flow between models and dae/models
*   Use SymPy/SynEngine to generate function calls - define the interfaces
*   Pickle/dill the auto-generated function calls on the first run (set dill recursive to True)
*   Use SymEngine to get the derivative for each model; the derivates may store in a smaller matrix locally to 
the model

*   Prototype Connectivity checking, previously implemented as `gyisland`
*   Initial values, pass initialized values between models, and solve initializer equations
# TODO: function for providing the jacobian triplet

# TODO: implement a standalone PI controller with numerical jacobians