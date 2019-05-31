## Modular Refactorization

*   Varout to DataFrame
*   Optimize `_varname()` in models

## Performance Improvement
*   Reduce the overhead in `VarOut.store()` and `TDS.dump_results()`
    *   Impact: Low
*   Reduce the overhead in `DAE.reset_Ac()`
    *   Impact: High