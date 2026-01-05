# Example: Dynamic Model (BusFreq)

This example walks through the real `BusFreq` model from ANDES to explain dynamic model concepts.

## Background

Dynamic models participate in time-domain simulation (TDS) and contribute both differential and
algebraic equations to the DAE system. They have state variables whose time derivatives define
system dynamics.

This tutorial examines the `BusFreq` model—a bus frequency measurement device. It demonstrates
key concepts of dynamic modeling: transfer function blocks, external variable coupling, and
signal processing. Unlike controllers that write back to other devices, this model reads bus
voltage angle and computes local frequency.

```{seealso}
Before proceeding, ensure you understand:
- {doc}`example-static` - Static model basics (start here if new)
- {doc}`../concepts/atoms` - v-providers and e-providers
- {doc}`../concepts/dae-formulation` - Differential equations in the DAE system
- {doc}`../components/blocks` - Transfer function blocks
```

## The BusFreq Model

The complete model is located at `andes/models/measurement/busfreq.py`. Here it is in full:

```python
# File: andes/models/measurement/busfreq.py

"""
Bus frequency estimation based on high-pass filter.
"""

from andes.core import (ModelData, Model, IdxParam, NumParam,
                        ConstService, ExtService, Algeb, ExtAlgeb,
                        Lag, Washout)


class BusFreq(ModelData, Model):
    """
    Bus frequency measurement. Outputs frequency in per unit value.

    The bus frequency output variable is `f`.
    The frequency deviation variable is `WO_y`.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'FreqMeasurement'

        # Parameters
        self.bus = IdxParam(info="bus idx", mandatory=True)

        self.Tf = NumParam(default=0.02,
                           info="input digital filter time const",
                           unit="sec",
                           tex_name='T_f',
                           )
        self.Tw = NumParam(default=0.1,
                           info="washout time const",
                           unit="sec",
                           tex_name='T_w',
                           )
        self.fn = NumParam(default=60.0,
                           info="nominal frequency",
                           unit='Hz',
                           tex_name='f_n',
                           )

        # Variables
        self.iwn = ConstService(v_str='u / (2 * pi * fn)', tex_name=r'1/\omega_n')
        self.a0 = ExtService(src='a',
                             model='Bus',
                             indexer=self.bus,
                             tex_name=r'\theta_0',
                             info='initial phase angle',
                             )
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          )
        self.L = Lag(u='(a-a0)',
                     T=self.Tf,
                     K=1,
                     info='digital filter',
                     )
        # the output `WO_y` is the frequency deviation in p.u.
        self.WO = Washout(u=self.L_y,
                          K=self.iwn,
                          T=self.Tw,
                          info='angle washout',
                          )
        self.WO_y.info = 'frequency deviation'
        self.WO_y.unit = 'p.u. (Hz)'

        self.f = Algeb(info='frequency output',
                       unit='p.u. (Hz)',
                       tex_name='f',
                       v_str='1',
                       e_str='1 + WO_y - f',
                       )
```

## Step-by-Step Breakdown

### Step 1: Class Structure

Unlike the Shunt model, BusFreq combines data and model in a single class:

```python
class BusFreq(ModelData, Model):
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.tds = True          # Enable for time-domain simulation
        self.group = 'FreqMeasurement'
```

Key observations:
- **`flags.tds = True`**: Marks this as a dynamic model for TDS
- **`group`**: Assigns to `FreqMeasurement` group
- No `flags.pflow` because frequency measurement is not meaningful in steady-state

### Step 2: Parameters

```python
        # Connection
        self.bus = IdxParam(info="bus idx", mandatory=True)

        # Time constants
        self.Tf = NumParam(default=0.02, info="input digital filter time const", unit="sec")
        self.Tw = NumParam(default=0.1, info="washout time const", unit="sec")

        # Nominal frequency
        self.fn = NumParam(default=60.0, info="nominal frequency", unit='Hz')
```

- **`bus`**: References which bus to measure
- **`Tf`**: Low-pass filter time constant for noise rejection
- **`Tw`**: Washout filter time constant for frequency extraction
- **`fn`**: Nominal system frequency (60 Hz in North America)

### Step 3: Services

```python
        self.iwn = ConstService(v_str='u / (2 * pi * fn)', tex_name=r'1/\omega_n')
        self.a0 = ExtService(src='a', model='Bus', indexer=self.bus,
                             tex_name=r'\theta_0', info='initial phase angle')
```

- **`ConstService`**: Calculates inverse of angular frequency: $\frac{1}{\omega_n} = \frac{1}{2\pi f_n}$
- **`ExtService`**: Captures the initial bus angle from power flow solution

The `u` in `v_str='u / (2 * pi * fn)'` is the online status flag (inherited from Model).
When `u=0` (offline), the service evaluates to zero.

### Step 4: External Variables

```python
        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta')
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name=r'V')
```

- **`ExtAlgeb`**: Links to algebraic variables from the Bus model
- **`a`**: Bus voltage angle (radians)
- **`v`**: Bus voltage magnitude (per unit)

Unlike the Shunt model, BusFreq does not inject power—it only reads the bus angle.

### Step 5: Transfer Function Blocks

This is where dynamic behavior is defined:

```python
        self.L = Lag(u='(a-a0)', T=self.Tf, K=1, info='digital filter')
        self.WO = Washout(u=self.L_y, K=self.iwn, T=self.Tw, info='angle washout')
```

**Lag Block**

The `Lag` block implements a first-order low-pass filter:

$$\frac{y}{u} = \frac{K}{1 + sT}$$

- **Input**: `(a - a0)` — angle deviation from initial value
- **Output**: `L_y` — filtered angle deviation
- **Purpose**: Removes high-frequency noise from angle measurement

**Washout Block**

The `Washout` block extracts rate of change (high-pass filter):

$$\frac{y}{u} = \frac{sKT}{1 + sT}$$

- **Input**: `L_y` — filtered angle deviation
- **Output**: `WO_y` — frequency deviation in per unit
- **Purpose**: Converts angle to frequency ($\omega = \frac{d\theta}{dt}$)

The blocks create internal state variables automatically. ANDES uses naming conventions:
- `L_y` is the output of block `L`
- `WO_y` is the output of block `WO`

### Step 6: Output Variable

```python
        self.f = Algeb(info='frequency output', unit='p.u. (Hz)', tex_name='f',
                       v_str='1',
                       e_str='1 + WO_y - f')
```

The final frequency output:
- **`v_str='1'`**: Initialize to 1.0 per unit (nominal frequency)
- **`e_str='1 + WO_y - f'`**: Equation $0 = 1 + \Delta\omega - f$, giving $f = 1 + \Delta\omega$

## Signal Flow

The signal processing chain:

```
Bus angle (a) → Subtract initial (a - a0) → Lag filter → Washout → Add 1.0 → f
                                             ↓              ↓
                                           L_y            WO_y
```

1. Read bus angle `a` from the Bus model
2. Subtract initial angle `a0` to get angle deviation
3. Apply low-pass filter (`Lag`) for noise rejection
4. Apply washout to get frequency deviation (derivative of angle)
5. Add nominal frequency (1.0 p.u.) to get absolute frequency

## Key Concepts Illustrated

### 1. Transfer Function Blocks

Blocks encapsulate state variables and equations:

```python
self.L = Lag(u='(a-a0)', T=self.Tf, K=1)
```

Internally, this creates:
- A state variable for the filter state
- Differential equation: $\dot{x} = \frac{K \cdot u - x}{T}$
- Output: `L_y = x`

### 2. Block Chaining

Blocks can be chained by referencing outputs:

```python
self.L = Lag(u='(a-a0)', ...)
self.WO = Washout(u=self.L_y, ...)  # L_y is output of L
```

### 3. Dynamic vs Static Flags

```python
self.flags.tds = True   # Include in time-domain simulation
# No flags.pflow        # Not included in power flow
```

### 4. External Services for Initialization

```python
self.a0 = ExtService(src='a', model='Bus', indexer=self.bus)
```

Captures the power flow solution value for use during TDS. Without this, the model
wouldn't know the initial bus angle.

## Testing the Model

```python
import andes

ss = andes.load('kundur_full.xlsx', setup=False)

# Add BusFreq to bus 1
ss.add('BusFreq', {'bus': 1, 'Tf': 0.02, 'Tw': 0.1})

ss.setup()
ss.PFlow.run()
ss.TDS.config.tf = 10
ss.TDS.run()

# Plot frequency measurement
ss.TDS.plt.plot(ss.BusFreq.f)
```

## Key Differences from Static Model

| Aspect | Static Model (Shunt) | Dynamic Model (BusFreq) |
|--------|---------------------|------------------------|
| Flags | `pflow=True, tds=True` | `tds=True` only |
| Behavior | Algebraic equations | Transfer function blocks |
| State | None | Internal block states |
| Purpose | Power injection | Signal processing |
| External vars | Write to bus (e_str) | Read only |

## See Also

- {doc}`model-structure` - General structure
- {doc}`example-static` - Static model example (Shunt)
- {doc}`example-tgov1` - Governor model with explicit state variables
- {doc}`example-ieeest` - Advanced controller with signal switching
- {doc}`../components/blocks` - Transfer function blocks reference
- Source code: `andes/models/measurement/busfreq.py`
