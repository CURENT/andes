# Modeling Concepts

Foundational concepts for understanding the ANDES modeling framework.

````{only} html
## Key Ideas

### Hybrid Symbolic-Numeric Framework
ANDES uses SymPy to define model equations symbolically, then generates optimized numerical code automatically. This enables:
- Automatic Jacobian derivation
- Self-documenting models with LaTeX equations
- Rapid prototyping of new models

### Atomic Types
Models are built from three fundamental atom types: parameters, variables, and services. These share a common **v-provider** interface (the `v` attribute) that enables interoperability in equations. Variables additionally serve as **e-providers**, contributing equation residuals to the DAE system.

### DAE Formulation
Power system dynamics are modeled as differential-algebraic equations:
- **Differential equations** (f): Generator dynamics, controller states
- **Algebraic equations** (g): Network power balance, algebraic constraints

### System Architecture
The `System` class orchestrates models, routines, and the DAE arrays.
````

```{toctree}
:maxdepth: 1

framework-overview
atoms
system-architecture
dae-formulation
```
