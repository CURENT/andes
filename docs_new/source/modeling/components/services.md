# Services

Services are helper variables outside the DAE variable list. They are most often
used for storing intermediate constants but can be used for special operations
to work around restrictions in the symbolic framework.

## Background

Services are **v-providers**, meaning each service has an attribute `v` for
storing service values. Unlike variables, services do not participate in the DAE
system directly but provide computed values that variables and equations can use.

The base class of services is {py:class}`andes.core.service.BaseService`.

## Service Types

```{eval-rst}
.. currentmodule:: andes.core.service
.. autosummary::
   :recursive:
   :toctree: _generated

   BaseService
   OperationService
   ConstService
   VarService
   ExtService
   PostInitService
   NumReduce
   NumRepeat
   IdxRepeat
   BackRef
   RefFlatten
   EventFlag
   VarHold
   ExtendedEvent
   DataSelect
   NumSelect
   DeviceFinder
   FlagCondition
   FlagValue
   FlagGreaterThan
   FlagLessThan
   InitChecker
   Replace
   ApplyFunc
```

| Class | Description |
|-------|-------------|
| `ConstService` | Internal service for constant values |
| `VarService` | Variable service updated at each iteration before equations |
| `ExtService` | External service for retrieving values from value providers |
| `PostInitService` | Constant service evaluated after TDS initialization |
| `NumReduce` | Reduce linear 2-D arrays into 1-D arrays |
| `NumRepeat` | Repeat a 1-D array to linear 2-D arrays |
| `IdxRepeat` | Repeat a 1-D list to linear 2-D list |
| `EventFlag` | Flag changes in inputs as an event |
| `VarHold` | Hold input value when a hold signal is active |
| `ExtendedEvent` | Extend an event signal for a given period of time |
| `DataSelect` | Select optional str data if provided or use the fallback |
| `NumSelect` | Select optional numerical data if provided |
| `DeviceFinder` | Find or create devices linked to the given devices |
| `BackRef` | Collect idx-es for backward references |
| `RefFlatten` | Convert BackRef list of lists into a 1-D list |
| `InitChecker` | Check initial values against typical values |
| `FlagValue` | Flag values that equal the given value |
| `Replace` | Replace values that return True for the given lambda func |

## Internal Constants

The most commonly used service is `ConstService`. It stores an array of constants
whose value is evaluated from a provided symbolic string. Constants are only
evaluated once in the model initialization phase, ahead of variable initialization.
`ConstService` is handy for calculating intermediate constants from parameters.

For example, a turbine governor has a `NumParam` `R` for the droop. `ConstService`
allows calculating the inverse of the droop (the gain) and using it in equations:

```python
self.R = NumParam()
self.G = ConstService(v_str='u/R')
```

where `u` is the online status parameter. The model can then use `G` in subsequent
variable or equation strings.

```{eval-rst}
.. autoclass:: andes.core.service.ConstService
   :noindex:
```

### VarService

Updated at each iteration before equation evaluation. Use for intermediate values
that depend on the current state of variables.

```{eval-rst}
.. autoclass:: andes.core.service.VarService
   :noindex:
```

### PostInitService

Evaluated after TDS initialization, when variable values have been determined.

```{eval-rst}
.. autoclass:: andes.core.service.PostInitService
   :noindex:
```

## External Constants

Service constants whose value is retrieved from an external model or group.
Using `ExtService` is similar to using external variables. The values of
`ExtService` will be retrieved once during the initialization phase before
`ConstService` evaluation.

For example, a synchronous generator needs to retrieve the `p` and `q` values
from static generators for initialization. In the `__init__()` of a synchronous
generator model, one can define:

```python
self.p0 = ExtService(src='p',
                     model='StaticGen',
                     indexer=self.gen,
                     tex_name='P_0')
```

```{eval-rst}
.. autoclass:: andes.core.service.ExtService
   :noindex:
```

## Shape Manipulators

This section is for advanced model developers.

All generated equations operate on 1-dimensional arrays and can use algebraic
calculations only. In some cases, one model would use `BackRef` to retrieve
2-dimensional indices and use such indices to retrieve variable addresses.
The retrieved addresses usually have a different length than the referencing
model and cannot be used directly for calculation.

Shape manipulator services can be used in such cases:

- `NumReduce` reduces a linearly stored 2-D ExtParam into 1-D Service
- `NumRepeat` repeats a 1-D value into linearly stored 2-D value based on the shape from a `BackRef`

```{eval-rst}
.. autoclass:: andes.core.service.BackRef
   :noindex:

.. autoclass:: andes.core.service.NumReduce
   :noindex:

.. autoclass:: andes.core.service.NumRepeat
   :noindex:

.. autoclass:: andes.core.service.IdxRepeat
   :noindex:

.. autoclass:: andes.core.service.RefFlatten
   :noindex:
```

## Value Manipulation

```{eval-rst}
.. autoclass:: andes.core.service.Replace
   :noindex:

.. autoclass:: andes.core.service.ParamCalc
   :noindex:

.. autoclass:: andes.core.service.ApplyFunc
   :noindex:
```

## Idx and References

```{eval-rst}
.. autoclass:: andes.core.service.DeviceFinder
   :noindex:
```

## Events

```{eval-rst}
.. autoclass:: andes.core.service.EventFlag
   :noindex:

.. autoclass:: andes.core.service.ExtendedEvent
   :noindex:

.. autoclass:: andes.core.service.VarHold
   :noindex:
```

## Flags

```{eval-rst}
.. autoclass:: andes.core.service.FlagCondition
   :noindex:

.. autoclass:: andes.core.service.FlagGreaterThan
   :noindex:

.. autoclass:: andes.core.service.FlagLessThan
   :noindex:

.. autoclass:: andes.core.service.FlagValue
   :noindex:
```

## Data Select

```{eval-rst}
.. autoclass:: andes.core.service.DataSelect
   :noindex:

.. autoclass:: andes.core.service.NumSelect
   :noindex:
```

## Miscellaneous

```{eval-rst}
.. autoclass:: andes.core.service.InitChecker
   :noindex:

.. autoclass:: andes.core.service.CurrentSign
   :noindex:

.. autoclass:: andes.core.service.RandomService
   :noindex:

.. autoclass:: andes.core.service.SwBlock
   :noindex:
```

## Common Patterns

### Per-Unit Conversion

```python
self.Zbase = ConstService(v_str='Vn**2 / Sn')
self.R_pu = ConstService(v_str='R_ohm / Zbase')
```

### Inverse Calculation

```python
self.R = NumParam()
self.gain = ConstService(v_str='u / R')
```

### Conditional Value

```python
self.K = ConstService(v_str='K1 * (mode == 1) + K2 * (mode == 2)')
```

## See Also

- {doc}`../concepts/atoms` - v-provider concept
- {doc}`parameters` - Input parameters
- {doc}`variables` - DAE variables
- {doc}`discrete` - Discrete components
