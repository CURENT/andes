# Hybrid Symbolic-Numeric Framework

ANDES uses a unique hybrid symbolic-numeric framework that combines the expressiveness of symbolic mathematics with the efficiency of numerical computation.

## Key Concept

Instead of writing numerical code directly, you **describe models using equations and building blocks**. ANDES then:

1. Parses symbolic equation strings
2. Computes partial derivatives automatically
3. Generates optimized numerical code
4. Executes vectorized calculations

## Benefits

### For Model Developers

- **Write equations, not code**: Define differential equations as strings like `'omega - 1'`
- **Automatic Jacobians**: No need to derive and code partial derivatives
- **Modular blocks**: Reuse transfer functions (Lag, LeadLag, PIController)
- **Built-in validation**: Framework catches common modeling errors

### For Users

- **Identical models**: PSS/E parameters work directly without conversion
- **Fast simulation**: Generated code is optimized and vectorized
- **Extensible**: Add new models without modifying core solver

## Framework Components

```
┌─────────────────────────────────────────────────┐
│                    Model Definition             │
│   Parameters  Variables  Equations  Blocks      │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Symbolic Processing                │
│   Parse → Derive → Optimize → Generate          │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Numerical Simulation               │
│   Initialize → Solve → Iterate → Output         │
└─────────────────────────────────────────────────┘
```

## Code Generation

When you first run ANDES or modify models, code generation occurs:

```bash
# Triggered automatically, or manually with:
andes prepare
```

This creates optimized Python code stored in `~/.andes/pycode/`. Generation runs once unless models change.

### What Gets Generated

- Equation evaluation functions
- Jacobian matrix builders
- Initialization routines
- Variable indexing maps

## Example: Model Definition

A simplified turbine governor in ANDES:

```python
class TGOV1(TGOVData, Model):
    def __init__(self, system, config):
        TGOVData.__init__(self)
        Model.__init__(self, system, config)

        # Services (intermediate constants)
        self.gain = ConstService(v_str='u/R')

        # Variables with equation strings
        self.pout = Algeb(
            v_str='tm0',
            e_str='(paux + pref + gain * (omega - 1) - pout)'
        )

        # Transfer function block
        self.LG = Lag(u=self.pout, T=self.T1, K=1)
```

Key observations:
- `v_str` defines initial value calculation
- `e_str` defines the algebraic equation (set to zero)
- `Lag` block encapsulates first-order dynamics
- No numerical code for derivatives—handled automatically

## Symbolic Variables

Within equation strings, you can reference:

| Type | Example | Description |
|------|---------|-------------|
| Parameters | `R`, `T1` | Model parameters |
| Variables | `omega`, `pout` | State/algebraic variables |
| External | `omega` | Variables from other models |
| Services | `gain` | Intermediate calculations |
| Block outputs | `LG_y` | Output of transfer blocks |

## Integration with Solver

During simulation:

1. **System** collects all model equations
2. **Equations** are evaluated using generated code
3. **Jacobians** are built incrementally by model
4. **Solver** finds the solution iteratively
5. **Results** are distributed back to models

## See Also

- {doc}`atoms` - v-provider and e-provider concepts
- {doc}`system-architecture` - System class internals
- {doc}`dae-formulation` - DAE mathematical background
- {doc}`../components/parameters` - Parameter types
- {doc}`../components/variables` - Variable types
