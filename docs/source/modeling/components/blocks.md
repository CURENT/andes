# Blocks

The block library contains commonly used blocks such as transfer functions and
nonlinear functions. Variables and equations are pre-defined for blocks to be
used as "lego pieces" for scripting DAE models.

## Background

The base class for blocks is {py:class}`andes.core.block.Block`.

All variables in a block must be defined as attributes in the constructor, just
like variable definition in models. The difference is that the variables are
"exported" from a block to the capturing model. All exported variables need to
be placed in a dictionary, `self.vars` at the end of the block constructor.

Blocks can be nested as advanced usage. See the API documentation below for
more details.

```{eval-rst}
.. autoclass:: andes.core.block.Block
   :members: define
   :noindex:
```

## Block Types

```{eval-rst}
.. currentmodule:: andes.core.block
.. autosummary::
   :recursive:
   :toctree: _generated

   Block
   Gain
   GainLimiter
   Piecewise
   HVGate
   LVGate
   DeadBand1
   Integrator
   IntegratorAntiWindup
   Lag
   LagAntiWindup
   LagFreeze
   LagAWFreeze
   LagRate
   LagAntiWindupRate
   Washout
   WashoutOrLag
   LeadLag
   LeadLagLimit
   Lag2ndOrd
   LeadLag2ndOrd
   PIController
   PIAWHardLimit
   PITrackAW
   PIFreeze
   PITrackAWFreeze
   PIDController
   PIDAWHardLimit
   PIDTrackAW
```

## Linear Blocks

```{eval-rst}
.. autoclass:: andes.core.block.Gain
   :members: define
   :noindex:

.. autoclass:: andes.core.block.GainLimiter
   :members: define
   :noindex:

.. autoclass:: andes.core.block.Piecewise
   :members: define
   :noindex:

.. autoclass:: andes.core.block.HVGate
   :noindex:

.. autoclass:: andes.core.block.LVGate
   :noindex:

.. autoclass:: andes.core.block.DeadBand1
   :noindex:
```

## First Order Blocks

```{eval-rst}
.. autoclass:: andes.core.block.Integrator
   :members: define
   :noindex:

.. autoclass:: andes.core.block.IntegratorAntiWindup
   :members: define
   :noindex:

.. autoclass:: andes.core.block.Lag
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LagAntiWindup
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LagFreeze
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LagAWFreeze
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LagRate
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LagAntiWindupRate
   :members: define
   :noindex:

.. autoclass:: andes.core.block.Washout
   :members: define
   :noindex:

.. autoclass:: andes.core.block.WashoutOrLag
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LeadLag
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LeadLagLimit
   :members: define
   :noindex:
```

## Second Order Blocks

```{eval-rst}
.. autoclass:: andes.core.block.Lag2ndOrd
   :members: define
   :noindex:

.. autoclass:: andes.core.block.LeadLag2ndOrd
   :members: define
   :noindex:
```

## PI Controllers

```{eval-rst}
.. autoclass:: andes.core.block.PIController
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PIAWHardLimit
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PITrackAW
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PIFreeze
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PITrackAWFreeze
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PIDController
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PIDAWHardLimit
   :members: define
   :noindex:

.. autoclass:: andes.core.block.PIDTrackAW
   :members: define
   :noindex:
```

## Saturation

```{eval-rst}
.. autoclass:: andes.models.exciter.ExcExpSat
   :members: define
   :noindex:
```

## Usage Examples

### First-Order Lag

```python
from andes.core.block import Lag

self.LG = Lag(
    u=self.x,   # Input
    T=self.T1,  # Time constant
    K=1         # Gain (default=1)
)
# Output: LG_y
```

### Washout Filter

```python
from andes.core.block import Washout

self.WO = Washout(
    u=self.x,
    T=self.Tw,
    K=self.Kw
)
# Output: WO_y
```

### Lead-Lag Compensator

```python
from andes.core.block import LeadLag

self.LL = LeadLag(
    u=self.x,
    T1=self.T1,  # Lead time constant
    T2=self.T2   # Lag time constant
)
# Output: LL_y
```

### PI Controller with Anti-Windup

```python
from andes.core.block import PIAWHardLimit

self.PIAW = PIAWHardLimit(
    u=self.error,
    kp=self.Kp,
    ki=self.Ki,
    lower=0,
    upper=1
)
# Output: PIAW_y
```

### Chaining Blocks

Blocks can be chained by using one block's output as another's input:

```python
# Input → Lag → LeadLag → Output
self.LG = Lag(u=self.input, T=self.T1, K=1)
self.LL = LeadLag(u=self.LG_y, T1=self.T2, T2=self.T3)
self.output = Algeb(e_str='LL_y - output')
```

## Naming Convention

We loosely follow a naming convention when using modeling blocks. An instance of
a modeling block is named with a two-letter acronym, followed by a number or a
meaningful but short variable name. The acronym and the name are spelled in
one word without underscore, as the output of the block already contains `_y`.

For example, two washout filters can be named `WO1` and `WO2`. In another
case, a first-order lag function for voltage sensing can be called `LGv`, or
even `LG` if there is only one Lag instance in the model.

| Block | Acronym | Example |
|-------|---------|---------|
| Lag | LG | LG, LGv, LG1 |
| Washout | WO | WO, WO1, WO2 |
| LeadLag | LL | LL, LL1, LLv |
| Integrator | IN | IN, INT |
| PIController | PI | PI, PIv |

Naming conventions are not strictly enforced. Expressiveness and concision are
encouraged.

## See Also

- {doc}`discrete` - Discrete components (many blocks use these internally)
- {doc}`variables` - Variable types
- {doc}`../creating-models/example-tgov1` - TGOV1 example using blocks
