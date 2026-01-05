# Testing Models

After creating a new model, thorough testing ensures correctness and robustness.

## Testing Levels

1. **Unit tests**: Model instantiation and basic behavior
2. **Integration tests**: Model works within full system
3. **Verification tests**: Results match reference implementations

## Quick Validation

### Check Model Loads

```python
import andes

ss = andes.System()

# Check model is registered
assert 'MyModel' in ss.models

# Check model documentation
print(ss.MyModel.doc())
```

### Check Parameters

```python
ss = andes.load('case.xlsx', setup=False)

# Add test device
ss.add('MyModel', {
    'bus': 1,
    'param1': 0.5,
    # ...
})

ss.setup()

# Verify parameters loaded
print(ss.MyModel.as_df())
```

## Power Flow Test

For models with `flags.pflow = True`:

```python
import andes

ss = andes.load('ieee14.xlsx')

# Run power flow
ss.PFlow.run()

# Check convergence
assert ss.PFlow.converged

# Check results are reasonable
v = ss.dae.y[ss.Bus.v.a]
assert all(v > 0.9) and all(v < 1.1)
```

## Time-Domain Test

For models with `flags.tds = True`:

### Flat Run

System should stay at equilibrium without disturbances:

```python
ss = andes.load('case.xlsx')
ss.PFlow.run()

# Run without disturbances
ss.TDS.config.tf = 2
ss.TDS.run()

# Check states stayed constant
omega = ss.dae.ts.x[:, ss.GENROU.omega.a]
deviation = omega.max() - omega.min()

assert deviation < 1e-6, "States should not change in flat run"
```

### Disturbance Response

```python
ss = andes.load('case.xlsx', setup=False)

# Add disturbance
ss.add('Fault', {'bus': 3, 'tf': 1.0, 'tc': 1.1})

ss.setup()
ss.PFlow.run()
ss.TDS.config.tf = 5
ss.TDS.run()

# Check simulation completed
assert ss.TDS.exit_code == 0

# Check reasonable response
omega_max = ss.dae.ts.x[:, ss.GENROU.omega.a].max()
assert omega_max < 1.1, "Speed deviation too large"
```

## Initialization Test

Check that initial conditions satisfy equations:

```python
ss = andes.load('case.xlsx')
ss.PFlow.run()
ss.TDS.init()

# Check equation residuals are small
f_max = abs(ss.dae.f).max()
g_max = abs(ss.dae.g).max()

assert f_max < 1e-6, f"Differential equations not satisfied: {f_max}"
assert g_max < 1e-6, f"Algebraic equations not satisfied: {g_max}"
```

## Writing pytest Tests

Create test file `tests/test_mymodel.py`:

```python
import pytest
import andes

class TestMyModel:
    """Tests for MyModel."""

    def test_instantiation(self):
        """Model can be instantiated."""
        ss = andes.System()
        assert 'MyModel' in ss.models

    def test_pflow(self):
        """Power flow converges with MyModel."""
        ss = andes.load('test_case.xlsx')
        ss.PFlow.run()
        assert ss.PFlow.converged

    def test_tds_flat(self):
        """TDS flat run is stable."""
        ss = andes.load('test_case.xlsx')
        ss.PFlow.run()
        ss.TDS.config.tf = 2
        ss.TDS.run()

        omega = ss.dae.ts.x[:, ss.GENROU.omega.a]
        assert omega.std() < 1e-6

    def test_tds_fault(self):
        """TDS with fault completes."""
        ss = andes.load('test_case.xlsx', setup=False)
        ss.add('Fault', {'bus': 3, 'tf': 1.0, 'tc': 1.1})
        ss.setup()
        ss.PFlow.run()
        ss.TDS.config.tf = 5
        ss.TDS.run()

        assert ss.TDS.exit_code == 0
```

Run tests:

```bash
pytest tests/test_mymodel.py -v
```

## Verification Against Reference

### Compare with PSS/E

1. Run same case in PSS/E
2. Apply identical disturbance
3. Export time series
4. Compare key variables

```python
import numpy as np
import pandas as pd

# Load ANDES results
andes_omega = ss.dae.ts.x[:, ss.GENROU.omega.a[0]]
andes_t = ss.dae.ts.t

# Load PSS/E results
psse = pd.read_csv('psse_results.csv')

# Interpolate to common time base
from scipy.interpolate import interp1d
f = interp1d(psse['time'], psse['omega'], fill_value='extrapolate')
psse_omega = f(andes_t)

# Compare
error = np.abs(andes_omega - psse_omega).max()
assert error < 0.01, f"Max error: {error}"
```

### Visual Comparison

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(andes_t, andes_omega, label='ANDES')
ax.plot(psse['time'], psse['omega'], '--', label='PSS/E')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Speed [pu]')
ax.legend()
plt.savefig('verification.png')
```

## Debugging Tips

### Check Variable Values

```python
# After init
print(f"omega initial: {ss.GENROU.omega.v}")
print(f"Pv initial: {ss.TGOVSimple.Pv.v}")
```

### Check Equation Residuals

```python
# After equation update
ss.f_update()
ss.g_update()

print(f"Max f residual: {abs(ss.dae.f).max()}")
print(f"Max g residual: {abs(ss.dae.g).max()}")
```

### Trace Jacobian Issues

```python
# Check Jacobian structure
ss.j_update()

import matplotlib.pyplot as plt
plt.spy(ss.dae.gy)
plt.title('Jacobian gy sparsity')
plt.savefig('jacobian.png')
```

### Step-by-Step Debugging

```python
ss.TDS.config.tf = 0.1  # Short simulation
ss.TDS.config.tstep = 0.001  # Small steps

# Enable verbose output
andes.config_logger(stream_level=10)  # DEBUG

ss.TDS.run()
```

## Common Issues

| Symptom | Possible Cause |
|---------|---------------|
| Initialization fails | Wrong `v_str` or circular dependency |
| Flat run drifts | Equation sign error |
| NaN during simulation | Division by zero, missing parameter |
| Jacobian singular | Missing equation or wrong `e_str` |
| Results differ from PSS/E | Parameter unit mismatch |

## See Also

- {doc}`model-structure` - Model structure reference
- {doc}`/verification/index` - Verification approach
