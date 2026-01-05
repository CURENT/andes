# Discrete Components

The discrete component library contains a special type of block for modeling
discontinuities in power system devices. Such discontinuities can be device-level
physical constraints or algorithmic limits imposed on controllers.

## Background

The base class for discrete components is {py:class}`andes.core.discrete.Discrete`.

The uniqueness of discrete components is the way they work. Discrete components
take inputs, criteria, and export a set of flags with component-defined meanings.
These exported flags can be used in algebraic or differential equations to build
piece-wise equations.

For example, `Limiter` takes a v-provider as input and two v-providers as the
upper and lower bounds. It exports three flags: `zi` (within bound), `zl` (below
lower bound), and `zu` (above upper bound).

### Update Timing

It is important to note when flags are updated. Discrete subclasses can use three
methods to check and update values and equations:

| Method | When Called | Purpose |
|--------|-------------|---------|
| `check_var` | Before equation evaluation | Variable-based flags (Limiter) |
| `check_eq` | After equation update | Equation-based flags (AntiWindup) |
| `set_var` | After solving | State pegging (AntiWindup) |

In the current implementation:
- `check_var` updates flags for variable-based discrete components (such as `Limiter`)
- `check_eq` updates flags for equation-involved discrete components (such as `AntiWindup`)
- `set_var` is currently only used by `AntiWindup` to store the pegged states

## Discrete Types

```{eval-rst}
.. currentmodule:: andes.core.discrete
.. autosummary::
   :recursive:
   :toctree: _generated

   Discrete
   Limiter
   SortedLimiter
   HardLimiter
   RateLimiter
   AntiWindup
   AntiWindupRate
   LessThan
   Selector
   Switcher
   DeadBand
   DeadBandRT
   Average
   Delay
   Derivative
   Sampling
   ShuntAdjust
```

## Limiters

```{eval-rst}
.. autoclass:: andes.core.discrete.Limiter
   :noindex:
```

Example usage:

```python
from andes.core.discrete import Limiter

self.lim = Limiter(
    u=self.x,        # Input variable
    lower=self.Vmin, # Lower bound
    upper=self.Vmax  # Upper bound
)

# Flags exported:
# lim.zi = 1 when lower <= x <= upper (within)
# lim.zl = 1 when x < lower (below)
# lim.zu = 1 when x > upper (above)

# Use in equation for piecewise behavior
self.y = Algeb(e_str='x*lim_zi + upper*lim_zu + lower*lim_zl - y')
```

See the code in `models/static/pq.py` for an example of voltage-based PQ-to-Z conversion.

```{eval-rst}
.. autoclass:: andes.core.discrete.SortedLimiter
   :noindex:

.. autoclass:: andes.core.discrete.HardLimiter
   :noindex:

.. autoclass:: andes.core.discrete.RateLimiter
   :noindex:

.. autoclass:: andes.core.discrete.AntiWindup
   :noindex:

.. autoclass:: andes.core.discrete.AntiWindupRate
   :noindex:
```

## Comparers

```{eval-rst}
.. autoclass:: andes.core.discrete.LessThan
   :noindex:

.. autoclass:: andes.core.discrete.Selector
   :noindex:

.. autoclass:: andes.core.discrete.Switcher
   :noindex:
```

Example of `Switcher` for multi-mode selection:

```python
from andes.core.discrete import Switcher

self.sw = Switcher(
    u=self.mode,    # Selection input
    options=[0, 1, 2]
)
# sw.s0 = 1 when mode == 0
# sw.s1 = 1 when mode == 1
# sw.s2 = 1 when mode == 2

# Use in equation
self.y = Algeb(e_str='K1*x*sw_s0 + K2*x*sw_s1 + K3*x*sw_s2 - y')
```

## Deadband

```{eval-rst}
.. autoclass:: andes.core.discrete.DeadBand
   :noindex:

.. autoclass:: andes.core.discrete.DeadBandRT
   :noindex:
```

Example:

```python
from andes.core.discrete import DeadBand

self.db = DeadBand(
    u=self.error,
    center=0,
    lower=-0.01,
    upper=0.01
)
# db.zi = 1 when within dead band
# db.zl = 1 when below
# db.zu = 1 when above

# Zero output in dead band
self.y = Algeb(e_str='error * (1 - db_zi) - y')
```

## Others

```{eval-rst}
.. autoclass:: andes.core.discrete.Average
   :noindex:

.. autoclass:: andes.core.discrete.Delay
   :noindex:

.. autoclass:: andes.core.discrete.Derivative
   :noindex:

.. autoclass:: andes.core.discrete.Sampling
   :noindex:

.. autoclass:: andes.core.discrete.ShuntAdjust
   :noindex:
```

## Using Flags in Equations

### Piecewise Linear

```python
# Saturated output
self.y = Algeb(
    e_str='x * lim_zi + upper * lim_zu + lower * lim_zl - y'
)
```

### Conditional Behavior

```python
# Different gains based on mode
self.y = Algeb(
    e_str='K1 * x * sw_s0 + K2 * x * sw_s1 - y'
)
```

### Dead Band Application

```python
# Zero output in dead band
self.y = Algeb(
    e_str='error * (1 - db_zi) - y'
)
```

## Naming Convention

Discrete component flags use the component name as prefix:

```python
self.lim = Limiter(...)
# Access in equations as: lim_zi, lim_zl, lim_zu

self.VL = Limiter(...)
# Access as: VL_zi, VL_zl, VL_zu
```

## Common Patterns

### Voltage-Based Load Shedding

```python
self.v_low = LessThan(u=self.v, bound=0.9)
self.P = Algeb(e_str='P0 * (1 - v_low_z) - P')
```

### Governor with Anti-Windup

```python
self.lim = AntiWindup(u=self.gate, lower=0, upper=1)
self.gate = State(e_str='...')  # Gate position
```

### Multi-Mode Controller

```python
self.mode = Switcher(u=self.ctrl_mode, options=[0, 1, 2])
self.output = Algeb(
    e_str='out1*mode_s0 + out2*mode_s1 + out3*mode_s2 - output'
)
```

## See Also

- {doc}`blocks` - Transfer function blocks (many use discrete internally)
- {doc}`services` - Service components
- {doc}`../concepts/dae-formulation` - Handling discontinuities in DAE
