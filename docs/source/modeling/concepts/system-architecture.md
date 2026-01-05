# System Architecture

The `System` class is the central orchestrator in ANDES, managing models, routines, and DAE data structures. The full API reference is found at {py:mod}`andes.system.System`.

## System Overview

```python
import andes

ss = andes.load('case.xlsx')
# ss is a System instance containing:
# - All loaded models (ss.Bus, ss.PQ, ss.GENROU, etc.)
# - Analysis routines (ss.PFlow, ss.TDS, ss.EIG)
# - DAE storage (ss.dae)
# - Configuration (ss.config)
```

## Dynamic Imports

System dynamically imports groups, models, and routines at creation. To add new
models, groups or routines, edit the corresponding registration file following
existing examples.

```
System
├── Groups (StaticLoad, Generator, Exciter, ...)
├── Models (Bus, PQ, GENROU, ESST1A, ...)
└── Routines (PFlow, TDS, EIG)
```

```{eval-rst}
.. autofunction:: andes.system.System.import_models
    :noindex:

.. autofunction:: andes.system.System.import_groups
    :noindex:

.. autofunction:: andes.system.System.import_routines
    :noindex:
```

## Code Generation

Under the hood, all models whose equations are provided in strings need to be
processed to generate executable functions for simulations. We call this process
"code generation". Code generation utilizes SymPy and can take up to one minute.

Code generation is automatically triggered upon the first ANDES run or whenever
model changes are detected. The generated code is stored and reused for speed up.

The generated Python code is called `pycode`. It is a Python package (folder)
with each module (a `.py` file) storing the executable Python code and metadata
for numerical simulation. The default storage path is `~/.andes/`.

```{note}
Code generation has been done if you have executed `andes`, `andes selftest`,
or `andes prepare`.
```

```{warning}
For developers: when models are modified (such as adding new models or changing
equation strings), code generation needs to be executed again for consistency.
ANDES can automatically detect changes, and it can be manually triggered from
command line using `andes prepare -i`.
```

```{eval-rst}
.. autofunction:: andes.system.System.prepare
    :noindex:

.. autofunction:: andes.system.System.undill
    :noindex:
```

## DAE Storage

The numerical DAE arrays are stored in `System.dae`:

```python
ss.dae.x    # State variables (differential)
ss.dae.y    # Algebraic variables
ss.dae.f    # Differential equations (dx/dt)
ss.dae.g    # Algebraic equations (= 0)
ss.dae.t    # Current time
```

### Time Series

After TDS, historical data is in:

```python
ss.dae.ts.t    # Time points
ss.dae.ts.x    # State history (n_steps x n_states)
ss.dae.ts.y    # Algebraic history (n_steps x n_algebs)
```

```{eval-rst}
.. autoclass:: andes.variables.dae.DAE
    :noindex:
```

## Decentralized Architecture

ANDES uses a decentralized architecture between models and DAE value arrays. In
this architecture, variables are initialized and equations are evaluated inside
each model. Then, `System` provides methods for collecting initial values and
equation values into `DAE`, as well as copying solved values back to each model.

1. **Models own their data**: Each model has local copies of variable values
2. **System orchestrates**: Collects/distributes values between models and DAE
3. **Parallel-friendly**: Model equations can be evaluated independently

### Data Flow

```
┌─────────┐         ┌─────────┐         ┌─────────┐
│  Model  │ ──────► │   DAE   │ ◄────── │  Model  │
│  (PQ)   │         │ arrays  │         │ (Gen)   │
└─────────┘         └─────────┘         └─────────┘
      │                  ▲                   │
      │                  │                   │
      └──────────────────┼───────────────────┘
                         │
                    ┌────┴────┐
                    │  System │
                    │ methods │
                    └─────────┘
```

The collection of values from models needs to follow protocols to avoid
conflicts. Details are given in the {doc}`../components/variables` section.

```{eval-rst}
.. autofunction:: andes.system.System.vars_to_dae
    :noindex:

.. autofunction:: andes.system.System.vars_to_models
    :noindex:

.. autofunction:: andes.system.System._e_to_dae
    :noindex:
```

## Jacobian Matrices

System builds sparse Jacobian matrices incrementally:

```python
ss.dae.fx    # df/dx (differential w.r.t. states)
ss.dae.fy    # df/dy (differential w.r.t. algebraic)
ss.dae.gx    # dg/dx (algebraic w.r.t. states)
ss.dae.gy    # dg/dy (algebraic w.r.t. algebraic)
```

### Matrix Sparsity Patterns

The largest overhead in building and solving nonlinear equations is the building
of Jacobian matrices. This is especially relevant when using the implicit
integration approach which algebraizes the differential equations. Given the
unique data structure of power system models, the sparse matrices for Jacobians
are built **incrementally**, model after model.

There are two common approaches to incrementally build a sparse matrix:

**Approach 1: In-place addition**
```python
self.fx += spmatrix(v, i, j, (n, n), 'd')
```
Although simple, this involves creating and discarding temporary objects on the
right hand side and, worse, changing the sparse pattern of `self.fx`.

**Approach 2: Collect and construct**
Store the rows, columns and values in array-like objects and construct the
Jacobians at the end. Very efficient but does not allow accessing the sparse
matrix while building.

**ANDES approach: Pre-allocation**
ANDES uses a pre-allocation approach to avoid changing sparse patterns by
filling values into a known sparse matrix pattern. System collects the indices
of rows and columns for each Jacobian matrix. Before in-place additions, ANDES
builds a temporary zero-filled `spmatrix`, to which the actual Jacobian values
are written later. Since these in-place add operations only modify existing
values, it does not change the pattern and thus avoids memory copying.

```{eval-rst}
.. autofunction:: andes.system.System.store_sparse_pattern
    :noindex:
```

## Calling Model Methods

System is an orchestrator for calling shared methods of models. These API
methods are defined for initialization, equation update, Jacobian update, and
discrete flags update.

The following methods take an argument `models`, which should be an
`OrderedDict` of models with names as keys and instances as values.

### Initialization

```{eval-rst}
.. autofunction:: andes.system.System.init
    :noindex:
```

### Equation Updates

```{eval-rst}
.. autofunction:: andes.system.System.e_clear
    :noindex:

.. autofunction:: andes.system.System.l_update_var
    :noindex:

.. autofunction:: andes.system.System.f_update
    :noindex:

.. autofunction:: andes.system.System.g_update
    :noindex:

.. autofunction:: andes.system.System.l_update_eq
    :noindex:

.. autofunction:: andes.system.System.j_update
    :noindex:
```

## Model Calling Protocol

System calls model methods in a defined order during each iteration:

1. `model.init()` - Initialize variables
2. `model.l_update_var()` - Update discrete flags (pre-equation)
3. `model.f_update()` - Evaluate differential equations
4. `model.g_update()` - Evaluate algebraic equations
5. `model.j_update()` - Update Jacobian contributions
6. `model.l_update_eq()` - Update discrete flags (post-equation)

## External Variable Protocol

When models share variables (e.g., Bus voltage accessed by loads):

| Flag | Purpose |
|------|---------|
| `v_setter=False` | Values at same address are summed |
| `v_setter=True` | This variable sets the final value |
| `e_setter=False` | Equation values are summed |
| `e_setter=True` | This equation sets the final value |

Example: PV generator sets bus voltage initial value:

```python
# In PV model
self.v = ExtAlgeb(src='v', model='Bus',
                  indexer=self.bus,
                  v_str='v0',
                  v_setter=True)  # Overwrite bus voltage
```

## Configuration

System, models and routines have a member attribute `config` for specific
configurations. System manages all configs, including saving to a config file
and loading back.

```python
ss.config              # System config
ss.PFlow.config        # Power flow config
ss.TDS.config          # TDS config
ss.GENROU.config       # Model-specific config
```

```{eval-rst}
.. autofunction:: andes.system.System.save_config
    :noindex:
```

```{warning}
Configs from files are passed to *model constructors* during instantiation. If
you need to modify config for a run, it must be done before instantiating
`System`, or before running `andes` from command line. Directly modifying
`Model.config` may not take effect or have side effects in the current
implementation.
```

## Device Lifecycle

Understanding how devices are created and managed helps when extending cases programmatically or developing new models.

### Data Loading Pipeline

When loading from files (xlsx, JSON, PSS/E), ANDES follows this sequence:

1. **File parsing**: I/O readers (`andes/io/xlsx.py`, `json.py`, `psse.py`) parse the input format into dictionaries
2. **Device registration**: Each row calls `System.add(model_name, param_dict)`:
   - Validates the model exists
   - Gets a unique `idx` from the device's group
   - Calls `Model.add()` to store parameter values
   - Registers the device with its group
3. **System setup**: `System.setup()` finalizes the structure (addresses, code generation)

```python
# This is what happens internally during andes.load():
# (simplified from andes/io/xlsx.py)
for name, df in df_models.items():
    for row in df.to_dict(orient='records'):
        system.add(name, row)  # Called once per device row
```

```{eval-rst}
.. autofunction:: andes.system.System.add
    :noindex:
```

### Programmatic Device Addition

Devices can be added programmatically before `setup()`:

```python
ss = andes.load('case.xlsx', setup=False)
ss.add('Fault', {'bus': 3, 'tf': 1.0, 'tc': 1.1})
ss.setup()
```

**Key constraints:**

- Devices cannot be added after `setup()` (raises `NotImplementedError`)
- Referenced devices must exist (e.g., `bus` must reference a valid `Bus.idx`)
- See {doc}`../components/parameters` for mandatory vs. optional parameters and per-unit conventions

## Timed Event Mechanism

Disturbance devices (`Fault`, `Toggle`, `Alter`) use `TimerParam` to schedule callbacks during time-domain simulation. These models belong to the `TimedEvent` group.

### How It Works

1. `TimerParam` stores a time value and a callback function
2. During TDS, the solver checks if `dae.t` matches any timer values
3. When triggered, the callback executes (e.g., `Fault.apply_fault()`)
4. The callback modifies system state (shunt admittance, device `u` flag, parameter values)

```python
# From andes/models/timer.py - Fault implementation
class Fault(ModelData, Model):
    def __init__(self, system, config):
        # ...
        self.tf = TimerParam(info='Bus fault start time',
                             callback=self.apply_fault)
        self.tc = TimerParam(info='Bus fault end time',
                             callback=self.clear_fault)

    def apply_fault(self, is_time: np.ndarray):
        """Apply fault when t = tf."""
        for i in range(self.n):
            if is_time[i] and self.u.v[i]:
                self.uf.v[i] = 1  # Enable fault equations
                # Store pre-fault algebraic variables for restoration
        return action
```

The fault equations inject a shunt admittance at the faulted bus:

```python
self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus,
                  e_str='u * uf * (v ** 2 * gf)')  # Active power
self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus,
                  e_str='-u * uf * (v ** 2 * bf)')  # Reactive power
```

### Available Timed Event Models

| Model | Purpose | Key Parameters |
|-------|---------|----------------|
| `Fault` | Three-phase fault at a bus | `bus`, `tf`, `tc`, `xf`, `rf` |
| `Toggle` | Switch device on/off | `model`, `dev`, `t` |
| `Alter` | Modify parameter value at runtime | `model`, `dev`, `src`, `t`, `method`, `amount` |

- **Fault**: Applies a shunt impedance (`xf`, `rf`) to ground at time `tf`, clears at `tc`
- **Toggle**: Negates the `u` (connectivity) field of any device at time `t`
- **Alter**: Applies arithmetic operations (`+`, `-`, `*`, `/`, `=`) to any parameter or service

For practical examples of adding disturbances, see {doc}`/tutorials/04-time-domain`.

## See Also

- {doc}`framework-overview` - Symbolic-numeric framework
- {doc}`atoms` - Value and equation providers
- {doc}`dae-formulation` - DAE mathematical details
