# Model Components

Building blocks for defining power system models.

```{seealso}
Before diving into specific components, understand the foundational concepts:
- {doc}`../concepts/atoms` - How parameters, variables, and services share a common interface (v-provider)
- {doc}`../concepts/dae-formulation` - How variables contribute to the DAE system (e-provider)
```

```{toctree}
:maxdepth: 1

parameters
variables
services
discrete
blocks
groups
```

## Component Types

### Parameters
Input data for models: `NumParam`, `IdxParam`, `ExtParam`, `DataParam`.

### Variables
State and algebraic variables: `State`, `Algeb`, `ExtState`, `ExtAlgeb`.

### Services
Intermediate calculations: `ConstService`, `VarService`, `ExtService`.

### Discrete Components
Non-smooth logic: `Limiter`, `DeadBand`, `Switcher`, `AntiWindup`.

### Transfer Function Blocks
Control system blocks: `Lag`, `LeadLag`, `Washout`, `PIController`, `Integrator`.

### Groups
Model classification and interface conventions.
