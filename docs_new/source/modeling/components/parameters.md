# Parameters

Parameters are a type of building atom for DAE models. Most parameters are read
directly from an input file and passed to equations, though some can be calculated
from existing parameters.

## Background

The base class for parameters in ANDES is `BaseParam`, which defines interfaces
for adding values and checking the number of values. `BaseParam` stores values
in a plain list via the member attribute `v`. Subclasses such as `NumParam`
store values using a NumPy ndarray for efficient numerical computation.

All parameters are **v-providers**: they have a `v` attribute containing values
that can be used in equation strings. This shared interface enables parameters
and services to be used interchangeably in equations. See {doc}`../concepts/atoms`
for details on the v-provider concept.

## Parameter Types

```{eval-rst}
.. currentmodule:: andes.core.param
.. autosummary::
   :recursive:
   :toctree: _generated

   BaseParam
   NumParam
   IdxParam
   ExtParam
   DataParam
   TimerParam
```

## NumParam

The most common parameter type for numerical values that participate in equations.

```python
from andes.core.param import NumParam

class MyModel(Model):
    def __init__(self):
        # Basic parameter
        self.R = NumParam(default=0.05,
                          info='Droop coefficient',
                          tex_name='R')

        # Parameter with constraints
        self.H = NumParam(default=3.0,
                          info='Inertia constant',
                          non_zero=True,
                          non_negative=True)

        # Parameter with units
        self.Vn = NumParam(default=20.0,
                           info='Rated voltage',
                           unit='kV')
```

`NumParam` values are stored in a NumPy array accessible via `self.R.v`. When
used in equation strings like `e_str='1/R'`, the parameter name is automatically
substituted with its values during evaluation.

```{eval-rst}
.. autoclass:: andes.core.param.NumParam
   :members:
   :noindex:
```

## IdxParam

References to devices in other models. `IdxParam` creates the connection graph
between models, enabling external parameters and variables to look up values
from referenced devices.

```python
from andes.core.param import IdxParam

class PQ(Model):
    def __init__(self):
        # Reference to Bus model
        self.bus = IdxParam(model='Bus',
                            mandatory=True,
                            info='Connected bus idx')
```

The connection graph enables model relationships:

```
GENROU.bus → Bus.idx
GENROU.gen → StaticGen.idx
ESST1A.syn → GENROU.idx
```

```{eval-rst}
.. autoclass:: andes.core.param.IdxParam
   :members:
   :noindex:
```

## ExtParam

Retrieves parameter values from external models. Uses an `IdxParam` as indexer
to look up values from the referenced model.

```python
from andes.core.param import ExtParam

class GENROU(Model):
    def __init__(self):
        # Get bus voltage from connected bus
        self.Vn = ExtParam(src='Vn',
                           model='Bus',
                           indexer=self.bus)

        # Get initial power from static generator
        self.p0 = ExtParam(src='p',
                           model='StaticGen',
                           indexer=self.gen)
```

`ExtParam` is essential for models that need configuration or initial conditions
from other models. The retrieved values are stored locally in a NumPy array and
can be used in equation strings just like `NumParam`.

```{eval-rst}
.. autoclass:: andes.core.param.ExtParam
   :members:
   :noindex:
```

## DataParam

For non-numeric data such as names or string identifiers that should not
participate in numerical calculations.

```python
from andes.core.param import DataParam

class Bus(Model):
    def __init__(self):
        self.name = DataParam(info='Bus name')
```

```{eval-rst}
.. autoclass:: andes.core.param.DataParam
   :members:
   :noindex:
```

## TimerParam

For time-based events. `TimerParam` stores callback functions that are triggered
when simulation time reaches the parameter value. This parameter type enables the
timed event mechanism used by `Fault`, `Toggle`, and `Alter` models.

```python
from andes.core.param import TimerParam

class Fault(Model):
    def __init__(self):
        self.tf = TimerParam(info='Fault start time',
                             callback=self.apply_fault)
        self.tc = TimerParam(info='Fault clear time',
                             callback=self.clear_fault)
```

The callback function receives a boolean array indicating which devices have
reached their trigger time. See {doc}`../concepts/system-architecture` for
details on how the timed event mechanism works internally.

```{eval-rst}
.. autoclass:: andes.core.param.TimerParam
   :members:
   :noindex:
```

## BaseParam

The abstract base class defining the parameter interface. All parameter types
inherit from `BaseParam`.

```{eval-rst}
.. autoclass:: andes.core.param.BaseParam
   :members:
   :noindex:
```

## Parameters in Equations

Parameters can be used directly in equation strings by name. The symbolic
framework substitutes parameter names with their `v` arrays during evaluation:

```python
self.R = NumParam(default=0.05)
self.D = NumParam(default=2.0)

# Use parameters in service calculation
self.gain = ConstService(v_str='1/R')

# Use parameters in differential equation
self.omega = State(e_str='Tm - Pe - D*(omega-1)')
```

## Mandatory vs Optional

- **Mandatory**: Must be provided in input data; simulation fails without it
- **Optional**: Uses default value if not provided

```python
self.bus = IdxParam(mandatory=True)   # Required
self.D = NumParam(default=0.0)        # Optional, defaults to 0
```

## Input Validation

ANDES validates parameters after loading:

- `NaN` values raise `ValueError`
- `inf` values are replaced with `1e8`
- Constraint violations (e.g., `non_zero`) use defaults with a warning

## See Also

- {doc}`../concepts/atoms` - v-provider and e-provider concepts
- {doc}`variables` - Variable types that use parameters in equations
- {doc}`services` - Service calculations using parameters
