# Variables

DAE Variables, or variables for short, are unknowns to be solved using numerical
or analytical methods. A variable stores values, equation values, and addresses
in the DAE array.

## Background

Variables are both **v-providers** and **e-providers** (see {doc}`../concepts/atoms`).
Each variable has member attributes `v` and `e` for variable values and equation
values, respectively. The initial value of `v` is set by the initialization routine,
and the initial value of `e` is set to zero.

During power flow calculation or time domain simulation:
- `v` is not directly modifiable by models but rather updated after solving non-linear equations
- `e` is updated by the models and summed up before solving equations

Each variable also stores addresses in its member attribute `a`. The addresses are
*0-based* indices into the numerical DAE array (`f` or `g`, based on variable type).

Equations associated with state variables take the form:

$$\mathbf{M} \dot{x} = \mathbf{f}(x, y)$$

where $\mathbf{x}$ are the differential variables, $\mathbf{y}$ are the algebraic
variables, $\mathbf{M}$ is the mass matrix, and $\mathbf{f}$ is the right-hand side
of differential equations.

Equations associated with algebraic variables take the form:

$$0 = \mathbf{g}(x, y)$$

where $\mathbf{g}$ is the equation right-hand side.

## Variable Types

```{eval-rst}
.. currentmodule:: andes.core.var
.. autosummary::
   :recursive:
   :toctree: _generated

   BaseVar
   State
   Algeb
   ExtVar
   ExtState
   ExtAlgeb
   AliasState
   AliasAlgeb
```

## State

State variables are described by differential equations and can only change
continuously. Use `State` for dynamics that evolve over time.

```python
from andes.core.var import State

class Generator(Model):
    def __init__(self):
        # Rotor angle
        self.delta = State(
            v_str='delta0',           # Initial value
            e_str='omega - 1',        # d(delta)/dt = omega - 1
            info='Rotor angle',
            tex_name=r'\delta'
        )

        # Rotor speed
        self.omega = State(
            v_str='1.0',
            e_str='(Tm - Te - D*(omega-1)) / M',
            info='Rotor speed',
            tex_name=r'\omega'
        )
```

```{eval-rst}
.. autoclass:: andes.core.var.State
   :members:
   :noindex:
```

## Algeb

Algebraic variables satisfy instantaneous constraints and can be discontinuous.
Use `Algeb` for power balance equations, output calculations, and constraints.

```python
from andes.core.var import Algeb

class Generator(Model):
    def __init__(self):
        # Electrical power output
        self.Pe = Algeb(
            v_str='p0',
            e_str='vd*Id + vq*Iq - Pe',
            info='Electrical power',
            tex_name='P_e'
        )
```

```{eval-rst}
.. autoclass:: andes.core.var.Algeb
   :members:
   :noindex:
```

## External Variables

Some models have "public" variables that can be accessed by other models. For
example, a `Bus` defines `v` for voltage magnitude. Each device attached to a
particular bus needs to access the value and impose reactive power injection.

External variables link with existing variables from another model or group using
`ExtAlgeb` or `ExtState`.

### ExtAlgeb

```python
from andes.core.var import ExtAlgeb

class PQ(Model):
    def __init__(self):
        # Access bus voltage
        self.v = ExtAlgeb(
            src='v',              # Source variable name
            model='Bus',          # Source model
            indexer=self.bus,     # IdxParam for lookup
            e_str='-q',           # Inject reactive power
        )
```

```{eval-rst}
.. autoclass:: andes.core.var.ExtAlgeb
   :members:
   :noindex:
```

### ExtState

```python
from andes.core.var import ExtState

class Exciter(Model):
    def __init__(self):
        # Access generator field voltage
        self.vf = ExtState(
            src='vf',
            model='GENROU',
            indexer=self.syn,
            e_str='Efd - vf0',
        )
```

```{eval-rst}
.. autoclass:: andes.core.var.ExtState
   :members:
   :noindex:
```

## Value and Equation Strings

The most important feature of the symbolic framework is allowing equations to be
defined using strings. There are three types of strings for a variable:

### v_str: Explicit Initialization

Equation string for **explicit** initialization in the form of `v = v_str(x, y)`.
The expression evaluates directly into the initial value.

```python
self.omega = State(v_str='1.0')  # Start at 1.0 pu
self.Pe = Algeb(v_str='p0')      # Start at initial power
```

### v_iter: Implicit Initialization

Equation string for **implicit** initialization in the form of `v_iter(x, y) = 0`.
All `v_iter` equations are solved numerically using the Newton-Krylov iterative method.

```python
self.Efd = Algeb(v_iter='Vf - Efd')  # Solve: Vf - Efd = 0
```

### e_str: Equation Definition

Equation string for the differential or algebraic equation:
- For `State`: right-hand side of $\dot{x} = f(x, y)$
- For `Algeb`: residual that should equal zero, $g(x, y) = 0$

```python
# Differential: d(omega)/dt = (Tm - Te) / M
self.omega = State(e_str='(Tm - Te) / M')

# Algebraic: 0 = P_gen - P_load
self.P = Algeb(e_str='Pgen - Pload')
```

## Flags for Value Overwriting

Variables have special flags for handling value initialization and equation values.
This is only relevant for public or external variables.

### v_setter Flag

The `v_setter` flag indicates whether a particular variable instance sets the initial
value. If `v_setter=False` (default), variable values of the same address are added.
If `v_setter=True`, the variable will set the values in the DAE array to its value,
overwriting any previous values.

Only one variable at the same address is allowed to have `v_setter=True`.

### e_setter Flag

The `e_setter` flag indicates whether the equation associated with a variable sets
the equation value rather than adding to it.

### A v_setter Example

A Bus is allowed to default the initial voltage magnitude to 1 and the voltage
phase angle to 0. If a PV device is connected to a Bus device, the PV should
override the voltage initial value with its voltage set point.

In `Bus.__init__()`:

```python
self.v = Algeb(v_str='1')
```

In `PV.__init__()`:

```python
self.v0 = Param()
self.bus = IdxParam(model='Bus')

self.v = ExtAlgeb(src='v',
                  model='Bus',
                  indexer=self.bus,
                  v_str='v0',
                  v_setter=True)
```

An `ExtAlgeb` is defined to access `Bus.v` using indexer `self.bus`. The `v_str`
sets the initial value to `v0`. During the variable initialization phase for `PV`,
`PV.v.v` is set to `v0`.

During value collection into `DAE.y` by the `System` class, `PV.v`, as a final
`v_setter`, will overwrite the voltage magnitude for Bus devices with the indices
provided in `PV.bus`.

## Alias Variables

Create references to existing variables for convenience:

```python
from andes.core.var import AliasState, AliasAlgeb

# Reference generator speed
self.omega = AliasState(self.GENROU.omega)
```

```{eval-rst}
.. autoclass:: andes.core.var.AliasState
   :members:
   :noindex:

.. autoclass:: andes.core.var.AliasAlgeb
   :members:
   :noindex:
```

## BaseVar

The abstract base class defining the variable interface.

```{eval-rst}
.. autoclass:: andes.core.var.BaseVar
   :members:
   :noindex:
```

## Common Patterns

### Power Balance

```python
# Algebraic equation: 0 = P_gen - P_load - P_line
self.P = Algeb(e_str='Pgen - Pload - Pline')
```

### Swing Equation

```python
self.delta = State(e_str='omega - 1')
self.omega = State(e_str='(Tm - Te - D*(omega-1)) / M')
```

### External Power Injection

```python
# Inject power to bus (negative = generation/injection)
self.a = ExtAlgeb(src='a', model='Bus', indexer=self.bus,
                  e_str='-p')
self.v = ExtAlgeb(src='v', model='Bus', indexer=self.bus,
                  e_str='-q')
```

## See Also

- {doc}`../concepts/atoms` - v-provider and e-provider concepts
- {doc}`../concepts/dae-formulation` - DAE mathematical background
- {doc}`parameters` - Parameter types
- {doc}`services` - Intermediate calculations
