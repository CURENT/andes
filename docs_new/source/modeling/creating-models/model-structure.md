# Model Structure

Every ANDES model follows a consistent structure with data definition, model logic, and registration.

```{seealso}
Understanding these foundational concepts will help:
- {doc}`../concepts/atoms` - How parameters, variables, and services work as v-providers and e-providers
- {doc}`../concepts/framework-overview` - The symbolic-numeric framework
```

## Basic Structure

A model consists of two classes: Data and Model.

```python
from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam
from andes.core.var import Algeb, State

class MyModelData(ModelData):
    """Data class defines parameters."""

    def __init__(self):
        super().__init__()
        # Define parameters here
        self.bus = IdxParam(model='Bus', mandatory=True)
        self.Sn = NumParam(default=100, info='Rated power')

class MyModel(MyModelData, Model):
    """Model class defines variables and equations."""

    def __init__(self, system, config):
        MyModelData.__init__(self)
        Model.__init__(self, system, config)

        # Model flags
        self.flags.pflow = True
        self.flags.tds = True

        # Group assignment
        self.group = 'StaticLoad'

        # Services, variables, equations here
```

## ModelData Class

Defines all input parameters:

```python
class MyModelData(ModelData):
    def __init__(self):
        super().__init__()

        # Required parameters
        self.bus = IdxParam(model='Bus', mandatory=True)

        # Numerical parameters with defaults
        self.p0 = NumParam(default=0, info='Active power')
        self.q0 = NumParam(default=0, info='Reactive power')

        # Voltage and power base
        self.Vn = NumParam(default=110, info='Rated voltage')
        self.Sn = NumParam(default=100, info='Rated power')
```

## Model Class

Defines services, variables, equations, and behaviors:

```python
class MyModel(MyModelData, Model):
    def __init__(self, system, config):
        MyModelData.__init__(self)
        Model.__init__(self, system, config)

        # 1. Flags - enable for specific analyses
        self.flags.pflow = True   # Power flow
        self.flags.tds = True     # Time-domain simulation
        self.flags.tds_init = True  # TDS initialization

        # 2. Group - for polymorphism
        self.group = 'StaticLoad'

        # 3. Services - intermediate calculations
        self.Zbase = ConstService(v_str='Vn**2/Sn')

        # 4. External parameters/variables
        self.v = ExtAlgeb(src='v', model='Bus', indexer=self.bus)

        # 5. Internal variables with equations
        self.P = Algeb(e_str='p0 - P')
```

## Model Flags

Control which analyses include this model:

| Flag | Analysis | Description |
|------|----------|-------------|
| `pflow` | Power flow | Include in PFlow equations |
| `tds` | Time-domain | Include in TDS equations |
| `tds_init` | TDS init | Custom TDS initialization |

```python
self.flags.pflow = True
self.flags.tds = True
```

## Component Order

Define components in this order for clarity:

1. **Flags and group**
2. **Config** (optional)
3. **ConstService** (constants)
4. **ExtService** (external constants)
5. **ExtParam** (external parameters)
6. **ExtVar** (external variables)
7. **VarService** (variable services)
8. **Discrete** (limiters, switches)
9. **Blocks** (transfer functions)
10. **Algeb** (algebraic variables)
11. **State** (state variables)

## Connecting to Other Models

### Reference via IdxParam

```python
# In data class
self.bus = IdxParam(model='Bus', mandatory=True)
self.gen = IdxParam(model='StaticGen', mandatory=True)
```

### Get External Parameters

```python
self.Vn = ExtParam(src='Vn', model='Bus', indexer=self.bus)
```

### Access External Variables

```python
# Algebraic variable from Bus
self.v = ExtAlgeb(
    src='v',
    model='Bus',
    indexer=self.bus,
    e_str='-q'  # Inject reactive power
)
```

## Equation Definition

### Algebraic Equation

$0 = g(x, y)$

```python
self.P = Algeb(
    v_str='p0',           # Initial value
    e_str='p0 - P',       # Equation: 0 = p0 - P
    info='Active power'
)
```

### Differential Equation

$\dot{x} = f(x, y)$

```python
self.omega = State(
    v_str='1.0',
    e_str='(Tm - Te) / M',  # d(omega)/dt = (Tm - Te) / M
    info='Rotor speed'
)
```

## Complete Example

```python
from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import ConstService

class PQData(ModelData):
    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus', mandatory=True)
        self.Vn = NumParam(default=110, info='Rated voltage [kV]')
        self.p0 = NumParam(default=0, info='Active power [pu]')
        self.q0 = NumParam(default=0, info='Reactive power [pu]')

class PQ(PQData, Model):
    def __init__(self, system, config):
        PQData.__init__(self)
        Model.__init__(self, system, config)

        self.flags.pflow = True
        self.flags.tds = True
        self.group = 'StaticLoad'

        # External bus voltage angle and magnitude
        self.a = ExtAlgeb(
            src='a', model='Bus', indexer=self.bus,
            e_str='-p0 * u',
            info='Bus angle'
        )
        self.v = ExtAlgeb(
            src='v', model='Bus', indexer=self.bus,
            e_str='-q0 * u',
            info='Bus voltage'
        )
```

## Registration

Add the model to `andes/models/__init__.py`:

```python
from andes.models.static.pq import PQ

# In the models dictionary
file_classes = OrderedDict([
    # ...
    ('pq', ['PQ']),
    # ...
])
```

## Code Generation

After creating or modifying a model:

```bash
andes prepare -i
```

This regenerates the numerical code for the new/modified model.

## See Also

- {doc}`example-static` - Static model walkthrough (Shunt)
- {doc}`example-dynamic` - Dynamic model walkthrough (BusFreq)
- {doc}`example-tgov1` - Advanced example with two implementation approaches
- {doc}`example-ieeest` - Comprehensive example with signal switching
- {doc}`testing-models` - Testing your models
