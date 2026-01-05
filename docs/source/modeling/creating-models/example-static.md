# Example: Static Model (Shunt)

This example walks through the real `Shunt` model from ANDES to explain static model concepts.

## Background

Static models participate in power flow analysis and contribute algebraic equations to the DAE system.
They do not have differential equations (state variables), making them simpler to implement than
dynamic models. Common examples include loads, shunts, and static VAR compensators.

This tutorial examines the `Shunt` model—a phasor-domain shunt compensator. It demonstrates the
constant impedance behavior where power varies with voltage squared ($P \propto V^2$, $Q \propto V^2$).

```{seealso}
Before proceeding, ensure you understand:
- {doc}`../concepts/atoms` - How parameters and variables work as v-providers and e-providers
- {doc}`../concepts/dae-formulation` - Algebraic equations in the DAE system
- {doc}`model-structure` - General model class structure
```

## The Shunt Model

The complete model is located at `andes/models/shunt/shunt.py`. Here it is in full:

```python
# File: andes/models/shunt/shunt.py

from andes.core import ModelData, IdxParam, NumParam, Model, ExtAlgeb


class ShuntData(ModelData):

    def __init__(self, system=None, name=None):
        super().__init__(system, name)

        self.bus = IdxParam(model='Bus', info="idx of connected bus", mandatory=True)

        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True, tex_name='S_n')
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True, tex_name='V_n')
        self.g = NumParam(default=0.0, info="shunt conductance (real part)", y=True, tex_name='g')
        self.b = NumParam(default=0.0, info="shunt susceptance (positive as capacitive)", y=True, tex_name='b')
        self.fn = NumParam(default=60.0, info="rated frequency", tex_name='f_n')


class ShuntModel(Model):
    """
    Shunt equations.
    """

    def __init__(self, system=None, config=None):
        Model.__init__(self, system, config)
        self.group = 'StaticShunt'
        self.flags.pflow = True
        self.flags.tds = True

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta',
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name='V',
                          ename='Q',
                          tex_ename='Q',
                          )

        self.a.e_str = 'u * v**2 * g'
        self.v.e_str = '-u * v**2 * b'


class Shunt(ShuntData, ShuntModel):
    """
    Phasor-domain shunt compensator Model.
    """

    def __init__(self, system=None, config=None):
        ShuntData.__init__(self)
        ShuntModel.__init__(self, system, config)
```

## Step-by-Step Breakdown

### Step 1: Data Class

The `ShuntData` class defines all input parameters:

```python
class ShuntData(ModelData):

    def __init__(self, system=None, name=None):
        super().__init__(system, name)

        # Connection to the network
        self.bus = IdxParam(model='Bus', info="idx of connected bus", mandatory=True)

        # Ratings
        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True, tex_name='S_n')
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True, tex_name='V_n')

        # Shunt admittance components
        self.g = NumParam(default=0.0, info="shunt conductance (real part)", y=True, tex_name='g')
        self.b = NumParam(default=0.0, info="shunt susceptance (positive as capacitive)", y=True, tex_name='b')

        self.fn = NumParam(default=60.0, info="rated frequency", tex_name='f_n')
```

Key observations:
- **`IdxParam`**: References another model (`Bus`) by index
- **`NumParam`**: Numerical parameters with defaults, units, and TeX names
- **`y=True`**: Marks parameters as part of the admittance matrix (for sparse solvers)
- **`mandatory=True`**: Parameter must be provided in input data

### Step 2: Model Class

The `ShuntModel` class defines the behavior:

```python
class ShuntModel(Model):

    def __init__(self, system=None, config=None):
        Model.__init__(self, system, config)

        # Group assignment for polymorphism
        self.group = 'StaticShunt'

        # Enable for power flow and time-domain simulation
        self.flags.pflow = True
        self.flags.tds = True
```

- **`group`**: Assigns to `StaticShunt` group for classification
- **`flags.pflow`**: Include in power flow equations
- **`flags.tds`**: Include in time-domain simulation

### Step 3: External Variables

Connect to bus voltage variables:

```python
        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta',
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name='V',
                          ename='Q',
                          tex_ename='Q',
                          )
```

- **`ExtAlgeb`**: External algebraic variable from another model
- **`model='Bus'`**: The source model
- **`src='a'`** / **`src='v'`**: The source variable name (angle / voltage magnitude)
- **`indexer=self.bus`**: Which bus instances to link to
- **`ename`** / **`tex_ename`**: Equation name for output

### Step 4: Power Injection Equations

The shunt admittance equations:

```python
        self.a.e_str = 'u * v**2 * g'
        self.v.e_str = '-u * v**2 * b'
```

These implement the constant impedance power equations:

$$P = V^2 \cdot g$$
$$Q = -V^2 \cdot b$$

Where:
- $g$ is conductance (positive = absorbs active power)
- $b$ is susceptance (positive = capacitive, generates reactive power)
- The negative sign on Q follows the generator convention (capacitor injects Q)
- **`u`** is the online status (inherited from `Model`)

### Step 5: Combined Class

The final class uses multiple inheritance:

```python
class Shunt(ShuntData, ShuntModel):
    """
    Phasor-domain shunt compensator Model.
    """

    def __init__(self, system=None, config=None):
        ShuntData.__init__(self)
        ShuntModel.__init__(self, system, config)
```

This pattern separates:
- **Data definition** (parameters) in the `*Data` class
- **Model behavior** (variables, equations) in the `*Model` class

## Key Concepts Illustrated

### 1. Parameters as v-providers

All parameters have a `v` attribute containing their values:

```python
# At runtime, for 3 shunt devices:
# self.g.v = np.array([0.01, 0.02, 0.015])
# self.b.v = np.array([0.05, 0.10, 0.08])
```

### 2. External variables as e-providers

`ExtAlgeb` links to variables in other models and contributes equations:

```python
self.a.e_str = 'u * v**2 * g'
```

When ANDES evaluates equations:
1. Substitutes `u`, `v`, `g` with their `.v` arrays (all are v-providers)
2. Stores result in `self.a.e` (equation contribution)
3. Accumulates into global DAE array at addresses `self.a.a`

### 3. Constant Impedance Behavior

The $V^2$ dependence is the defining characteristic:
- When voltage drops, power consumption drops quadratically
- This is more stable than constant power loads during voltage disturbances
- Critical for voltage stability studies

### 4. Sign Convention

- **Positive injection** = power flowing into the bus (generation)
- **Negative injection** = power flowing out of the bus (load)
- Capacitive shunt ($b > 0$): $Q = -V^2 b < 0$ means reactive power is injected (generated)

## Testing the Model

```python
import andes

ss = andes.load('ieee14.xlsx')

# Check existing shunts
print(ss.Shunt.as_df())

# Run power flow
ss.PFlow.run()

# Check reactive power injection
print(f"Shunt Q injection: {ss.Shunt.v.e}")
```

## See Also

- {doc}`model-structure` - General model structure
- {doc}`example-dynamic` - Dynamic model with states
- {doc}`../components/parameters` - Parameter types
- {doc}`../components/variables` - Variable types and the v-e-a triad
- Source code: `andes/models/shunt/shunt.py`
