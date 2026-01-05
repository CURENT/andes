# Example: IEEEST Power System Stabilizer

This comprehensive example walks through the complete implementation of the
IEEEST power system stabilizer model. It demonstrates advanced techniques
including optional parameters, input selection, external services, and mode
switching.

## Model Overview

IEEEST is a versatile PSS model with:
- Multiple input signal options (speed, frequency, power, voltage)
- Two stages of lead-lag compensation
- Gain and washout blocks
- Output limiting

## Block Diagram

```
                                                      LSMAX
                                                        │
Signal ──►[F1: 2nd-order]──►[F2: Lead-Lag 2nd]──►      ↓
  │           Lag              order           [LL1]──►[LL2]──►[×Ks]──►[WO]──►[Lim]──► Vss
  │                                                                              │
  │                                                                              ↓
  ▼                                                                            LSMIN
 MODE selects:
 1: ω-1 (speed deviation)
 2: f-1 (frequency deviation)
 3: Pe/SnSb (electrical power)
 4: Pm-Pm0 (accelerating power)
 5: V (bus voltage)
 6: dV/dt (voltage derivative)
```

## Data Classes

### PSSBaseData

`PSSBaseData` holds parameters shared by all PSS models:

```python
class PSSBaseData(ModelData):
    def __init__(self):
        super().__init__()
        # Link to exciter (required for all PSS)
        self.avr = IdxParam(model='Exciter',
                            mandatory=True,
                            info='Exciter idx')
```

### IEEESTData

`IEEESTData` defines IEEEST-specific parameters:

```python
class IEEESTData(PSSBaseData):
    def __init__(self):
        PSSBaseData.__init__(self)

        # Input mode selection
        self.MODE = NumParam(default=1,
                             info='Input signal mode (1-6)')

        # Optional remote bus
        self.busr = IdxParam(model='Bus',
                             info='Remote bus idx',
                             default=None)

        # Transfer function parameters
        self.A1 = NumParam(default=0, info='2nd-order filter coeff.')
        self.A2 = NumParam(default=0, info='2nd-order filter coeff.')
        # ... additional parameters
```

## PSSBase Model Class

`PSSBase` defines common external connections shared by all PSS models:

```python
class PSSBase(Model):
    def __init__(self, system, config):
        super().__init__(system, config)

        # Register with PSS group and enable TDS
        self.group = 'PSS'
        self.flags.tds = True
```

```{important}
Setting `self.flags.tds = True` is required to include the model in time-domain
simulation. Without this flag, data loads but variables receive no addresses
and equations are skipped.
```

### Parameter Replacement

The `Replace` service replaces input values that satisfy a condition:

```python
# Replace zero limits with large values (effectively no limit)
self.VCUr = Replace(self.VCU, lambda x: np.equal(x, 0.0), 999)
self.VCLr = Replace(self.VCL, lambda x: np.equal(x, 0.0), -999)
```

### Retrieving Connected Devices

PSS needs to find the generator connected through the exciter:

```python
# Get generator idx from exciter
self.syn = ExtParam(model='Exciter',
                    src='syn',
                    indexer=self.avr,
                    export=False,
                    info='Retrieved generator idx',
                    vtype=str)

# Get bus from generator
self.bus = ExtParam(model='SynGen',
                    src='bus',
                    indexer=self.syn,
                    export=False,
                    info='Retrieved bus idx',
                    vtype=str,
                    default=None)
```

### Optional Remote Bus

PSS models support an optional remote bus. `DataSelect` chooses between
the optional input and a fallback:

```python
# Use busr if provided, otherwise use local bus
self.buss = DataSelect(self.busr,
                       self.bus,
                       info='selected bus (bus or busr)')
```

### Device Finding

`DeviceFinder` locates or creates measurement devices:

```python
# Find or create frequency measurement device for the selected bus
self.busfreq = DeviceFinder(self.busf,
                            link=self.buss,
                            idx_name='bus')
```

If `busf` is not specified in the input data, `DeviceFinder` will find an
existing `BusFreq` device for the bus or create one.

## IEEESTModel

### Configuration

Add model-specific configuration options:

```python
self.config.add(OrderedDict([('freq_model', 'BusFreq')]))
self.config.add_extra('_help', {'freq_model': 'default freq. measurement model'})
self.config.add_extra('_alt', {'freq_model': ('BusFreq',)})

# Set the measurement model for DeviceFinder
self.busf.model = self.config.freq_model
```

### Voltage Derivative

For input mode 6, we need dV/dt. `Derivative` computes finite differences:

```python
self.dv = Derivative(self.v,
                     tex_name='dV/dt',
                     info='Finite difference of bus voltage')
```

### Per-Unit Conversion

Retrieve the machine-to-system base conversion factor:

```python
self.SnSb = ExtService(model='SynGen',
                       src='M',
                       indexer=self.syn,
                       attr='pu_coeff',
                       info='Machine base to sys base factor for power',
                       tex_name='(Sb/Sn)')
```

Note the `attr='pu_coeff'` - this accesses a special attribute of the `M`
variable that stores the per-unit conversion coefficient.

### Input Mode Selection with Switcher

`Switcher` parses the input mode into boolean flags:

```python
self.SW = Switcher(u=self.MODE,
                   options=[0, 1, 2, 3, 4, 5, 6])
```

This creates flags `SW_s0` through `SW_s6`. We include `0` for padding so
that `SW_s1` corresponds to MODE 1.

### Input Signal Selection

The input signal uses piece-wise construction based on mode:

```python
self.sig = Algeb(tex_name='S_{ig}',
                 info='Input signal')

# Initial value (uses MODE-dependent expression)
self.sig.v_str = ('SW_s1*(omega-1) + SW_s2*0 + SW_s3*(tm0/SnSb) + '
                  'SW_s4*(tm-tm0) + SW_s5*v + SW_s6*0')

# Equation (uses MODE-dependent expression)
self.sig.e_str = ('SW_s1*(omega-1) + SW_s2*(f-1) + SW_s3*(te/SnSb) + '
                  'SW_s4*(tm-tm0) + SW_s5*v + SW_s6*dv_v - sig')
```

**Key observations:**
- `v_str` and `e_str` are assigned separately for readability
- Each mode is multiplied by its corresponding flag (`SW_s1`, `SW_s2`, etc.)
- Only the active mode contributes to the sum

### Transfer Function Blocks

The signal processing chain uses nested blocks:

```python
# Second-order lag filter
self.F1 = Lag2ndOrd(u=self.sig, K=1, T1=self.A1, T2=self.A2)

# Second-order lead-lag
self.F2 = LeadLag2ndOrd(u=self.F1_y,
                        T1=self.A3, T2=self.A4,
                        T3=self.A5, T4=self.A6,
                        zero_out=True)

# First lead-lag stage
self.LL1 = LeadLag(u=self.F2_y, T1=self.T1, T2=self.T2, zero_out=True)

# Second lead-lag stage
self.LL2 = LeadLag(u=self.LL1_y, T1=self.T3, T2=self.T4, zero_out=True)

# Gain
self.Vks = Gain(u=self.LL2_y, K=self.KS)

# Washout or lag (depending on T5/T6)
self.WO = WashoutOrLag(u=self.Vks_y,
                       T=self.T6,
                       K=self.T5,
                       name='WO',
                       zero_out=True)
```

### Output Limiting

Two-stage limiting with algebraic variable:

```python
# Internal signal limiter
self.VLIM = Limiter(u=self.WO_y,
                    lower=self.LSMIN,
                    upper=self.LSMAX,
                    info='Vss limiter')

# Vss applies the limiter
self.Vss = Algeb(tex_name='V_{ss}',
                 info='Voltage output before output limiter',
                 e_str='VLIM_zi * WO_y + VLIM_zu * LSMAX + VLIM_zl * LSMIN - Vss')

# Output limiter based on bus voltage
self.OLIM = Limiter(u=self.v,
                    lower=self.VCLr,
                    upper=self.VCUr,
                    info='output limiter')

# Final output (defined in PSSBase, equation assigned here)
self.vsout.e_str = 'OLIM_zi * Vss - vsout'
```

## Final Assembly

Combine data and model classes:

```python
class IEEEST(IEEESTData, IEEESTModel):
    def __init__(self, system, config):
        IEEESTData.__init__(self)
        IEEESTModel.__init__(self, system, config)
```

## Model Registration

### Adding to Model List

Edit `andes/models/__init__.py`:

```python
file_classes = list([
    ...
    ('pss', ['IEEEST', 'ST2CUT']),
    ...
])
```

### Group Definition

Ensure the PSS group exists in `andes/models/group.py`:

```python
class PSS(GroupBase):
    """Power system stabilizer group."""

    def __init__(self):
        super().__init__()
        self.common_vars.extend(('vsout',))
```

All PSS models must have `vsout` as a common variable.

## Key Techniques Summary

| Technique | Component | Purpose |
|-----------|-----------|---------|
| `Replace` | Service | Substitute invalid input values |
| `ExtParam` | Parameter | Retrieve parameters from linked models |
| `DataSelect` | Service | Choose between optional and fallback values |
| `DeviceFinder` | Service | Find or create linked devices |
| `Derivative` | Discrete | Compute finite differences |
| `Switcher` | Discrete | Parse mode into boolean flags |
| `zero_out` | Block option | Output zero when time constant is zero |

## Complete Source Code

The complete implementation is in `andes/models/pss/ieeest.py`.

## See Also

- {doc}`example-tgov1` - Simpler example with two implementation styles
- {doc}`../components/services` - Services including DataSelect and DeviceFinder
- {doc}`../components/discrete` - Discrete components including Switcher
- {doc}`../components/blocks` - Transfer function blocks
