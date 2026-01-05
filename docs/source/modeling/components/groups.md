# Groups

A group is a collection of similar functional models with common variables and
parameters. It is mandatory to enforce the common variables and parameters when
developing new models. The common variables and parameters are typically the
interface when connecting different group models.

## Background

For example, the Group `RenGen` has variables `Pe` and `Qe`, which are active
power output and reactive power output. Such common variables can be retrieved
by other models, such as one in the Group `RenExciter` for further calculation.

In such a way, the same variable interface is realized so that all models in
the same group can carry out similar functions. This enables:

1. **Common interface**: All generators have `bus`, `Sn`, `Vn`
2. **Polymorphism**: Exciters work with any synchronous generator model
3. **Backward references**: Find all devices connected to a bus
4. **Model substitution**: Replace GENCLS with GENROU without changing case files

## GroupBase

```{eval-rst}
.. currentmodule:: andes.models.group
.. autosummary::
   :recursive:
   :toctree: _generated

   GroupBase
```

```{eval-rst}
.. autoclass:: andes.models.group.GroupBase
   :members:
   :noindex:
```

## Standard Groups

| Group | Models | Common Parameters |
|-------|--------|-------------------|
| `StaticGen` | PV, Slack | bus, p0, q0, Sn |
| `SynGen` | GENCLS, GENROU, GENSAL | bus, gen, Sn, Vn |
| `Exciter` | ESST1A, ESDC2, SEXS | syn |
| `TurbineGov` | TGOV1, IEEEG1, HYGOV | syn |
| `PSS` | IEEEST, ST2CUT | avr |
| `StaticLoad` | PQ, ZIP | bus, Vn |
| `Motor` | MotorThree, MotorFive | bus |
| `RenGen` | REGCA1, REECA1 | bus, gen |

## Defining Groups

Groups are defined in `andes/models/group.py`:

```python
from andes.core.model import GroupBase

class Exciter(GroupBase):
    """Exciter group base class."""

    def __init__(self):
        super().__init__()
        # Common parameters for all exciters
        self.common_params = ['syn']
        self.common_vars = []
```

## Model Group Registration

Models declare their group in the constructor:

```python
class ESST1A(ExciterData, Model):
    def __init__(self, system, config):
        ExciterData.__init__(self)
        Model.__init__(self, system, config)

        self.group = 'Exciter'  # Register with Exciter group
```

## Using Groups for References

### External Parameters

Get parameters from any model in a group:

```python
class TurbineGov(Model):
    def __init__(self):
        # Reference generator - can be any SynGen model
        self.syn = IdxParam(model='SynGen', mandatory=True)

        # Get inertia from the generator (works for GENCLS or GENROU)
        self.M = ExtParam(src='M', model='SynGen', indexer=self.syn)
```

### External Variables

Access variables from any group member:

```python
class Exciter(Model):
    def __init__(self):
        self.syn = IdxParam(model='SynGen')

        # Access generator terminal voltage
        self.vd = ExtAlgeb(src='vd', model='SynGen', indexer=self.syn)
        self.vq = ExtAlgeb(src='vq', model='SynGen', indexer=self.syn)
```

## BackRef for Connections

Groups enable backward references:

```python
class Bus(Model):
    def __init__(self):
        # Collect all PQ loads connected to this bus
        self.PQ = BackRef()

        # Collect all generators
        self.SynGen = BackRef()
```

Usage:

```python
# Get list of PQ indices on bus 1
pq_indices = ss.Bus.PQ.v[0]  # List of PQ idx connected to first bus
```

## Group Substitution

Replace models without changing input files:

```python
# Original case uses GENCLS
# To use GENROU instead, load with model map
ss = andes.load('case.xlsx', model_map={'GENCLS': 'GENROU'})
```

Since both are in `SynGen` group, references work automatically.

## Common Group Patterns

### Generator Groups

```
StaticGen (base)
    ├── PV
    └── Slack

SynGen (dynamic generators)
    ├── GENCLS (classical)
    ├── GENROU (round-rotor)
    └── GENSAL (salient-pole)
```

### Controller Groups

```
Exciter
    ├── ESST1A
    ├── ESDC2
    ├── SEXS
    └── ...

TurbineGov
    ├── TGOV1
    ├── IEEEG1
    ├── HYGOV
    └── ...
```

## Group Requirements

When creating models for a group:

1. **Inherit group data class**: Use `ExciterData`, `TGOVData`, etc.
2. **Set group name**: `self.group = 'Exciter'`
3. **Provide required parameters**: Match `common_params`
4. **Match interface**: Variables and equations must be compatible

## Creating New Groups

```python
# In andes/models/group.py
class MyGroup(GroupBase):
    def __init__(self):
        super().__init__()
        self.common_params = ['bus', 'Sn']
        self.common_vars = ['P', 'Q']
```

Register in the models import file to make it available.

## See Also

- {doc}`parameters` - Parameter types including IdxParam
- {doc}`variables` - External variables
- {doc}`services` - BackRef service
- {doc}`../creating-models/model-structure` - Complete model structure
