# Example: TGOV1 Turbine Governor

This example demonstrates modeling a turbine governor using both equation-based
and block-based approaches. TGOV1 is a commonly used turbine governor model that
is sufficiently complex to illustrate key modeling concepts.

## Model Overview

The TGOV1 turbine governor model consists of:
- A droop characteristic
- A first-order lag with anti-windup limiter
- A lead-lag transfer function

## Block Diagram

```
           ┌───────────────────────────────────────────────────────┐
           │                                                       │
  ωref ─+──│──►(×)───►┌──────┐    ┌───────────┐    ┌─────────┐    │
         │ │    1/R   │ Lag  │───►│ Lead-Lag  │───►│  + Dt   │────┼──► Pout
  ω ─────┼─┘          │ K/T1 │    │ (T2,T3)   │    └─────────┘    │
         │            │ VMIN │    └───────────┘         ▲         │
         │            │ VMAX │                          │         │
         │            └──────┘                          │         │
         └──────────────────────────────────────────────┘         │
                                                                  ▼
                                                                 tm
```

## Mathematical Equations

The model is described by two differential equations and six algebraic equations:

**Differential Equations:**

$$
\begin{bmatrix}
\dot{x}_{LG} \\
\dot{x}_{LL}
\end{bmatrix}
=
\begin{bmatrix}
z_{i,lim}^{LG} (P_{d} - x_{LG}) / T_1 \\
(x_{LG} - x_{LL}) / T_3
\end{bmatrix}
$$

**Algebraic Equations:**

$$
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix}
=
\begin{bmatrix}
(1 - \omega) - \omega_{d} \\
R \times \tau_{m0} - P_{ref} \\
(P_{ref} + \omega_{d})/R - P_{d} \\
D_t \omega_{d} + y_{LL} - P_{OUT} \\
\frac{T_2}{T_3}(x_{LG} - x_{LL}) + x_{LL} - y_{LL} \\
u(P_{OUT} - \tau_{m0})
\end{bmatrix}
$$

Where:
- $x_{LG}$, $x_{LL}$ are internal states of the lag and lead-lag blocks
- $y_{LL}$ is the lead-lag output
- $\omega$ is generator speed, $\omega_d$ is generator under-speed
- $P_d$ is droop output, $\tau_{m0}$ is steady-state torque input
- $P_{OUT}$ is turbine output summed at the generator

## Equation-Based Implementation

The following code implements TGOV1 using explicit equations. This approach
gives complete control over the mathematical formulation.

```python
def __init__(self, system, config):
    # 1. Declare parameters from case file inputs
    self.R = NumParam(info='Turbine governor droop',
                      non_zero=True, ipower=True)
    self.T1 = NumParam(info='Lag time constant')
    self.T2 = NumParam(info='Lead time constant')
    self.T3 = NumParam(info='Lag time constant')
    self.Dt = NumParam(info='Damping coefficient')
    self.VMIN = NumParam(info='Minimum gate limit')
    self.VMAX = NumParam(info='Maximum gate limit')

    # 2. Declare external variables from generators
    self.omega = ExtState(src='omega',
                          model='SynGen',
                          indexer=self.syn,
                          info='Generator speed')
    self.tm = ExtAlgeb(src='tm',
                       model='SynGen',
                       indexer=self.syn,
                       e_str='u*(pout-tm0)',
                       info='Generator torque input')

    # 3. Declare initial values from generators
    self.tm0 = ExtService(src='tm',
                          model='SynGen',
                          indexer=self.syn,
                          info='Initial torque input')

    # 4. Declare variables and equations
    self.pref = Algeb(info='Reference power input',
                      v_str='tm0*R',
                      e_str='tm0*R-pref')

    self.wd = Algeb(info='Generator under speed',
                    e_str='(1-omega)-wd')

    self.pd = Algeb(info='Droop output',
                    v_str='tm0',
                    e_str='(wd+pref)/R-pd')

    self.LG_x = State(info='State in the lag TF',
                      v_str='pd',
                      e_str='LG_lim_zi*(pd-LG_x)/T1')

    self.LG_lim = AntiWindup(u=self.LG_x,
                             lower=self.VMIN,
                             upper=self.VMAX)

    self.LL_x = State(info='State in the lead-lag TF',
                      v_str='LG_x',
                      e_str='(LG_x-LL_x)/T3')

    self.LL_y = Algeb(info='Lead-lag Output',
                      v_str='LG_x',
                      e_str='T2/T3*(LG_x-LL_x)+LL_x-LL_y')

    self.pout = Algeb(info='Turbine output power',
                      v_str='tm0',
                      e_str='(LL_y+Dt*wd)-pout')
```

**Key observations:**
- `ExtState` and `ExtAlgeb` link to generator variables
- `ExtService` retrieves initial values for initialization
- `AntiWindup` discrete component provides limiter flags
- The `LG_lim_zi` flag in `e_str` implements anti-windup behavior

## Block-Based Implementation

The same model can be implemented more concisely using transfer function blocks.
This approach improves readability and reduces the chance of equation errors.

```python
def __init__(self, system, config):
    TGBase.__init__(self, system, config)

    # Calculate gain from droop
    self.gain = ConstService(v_str='u/R')

    # Reference power
    self.pref = Algeb(info='Reference power input',
                      tex_name='P_{ref}',
                      v_str='tm0 * R',
                      e_str='tm0 * R - pref')

    # Generator under-speed
    self.wd = Algeb(info='Generator under speed',
                    unit='p.u.',
                    tex_name=r'\omega_{dev}',
                    v_str='0',
                    e_str='(wref - omega) - wd')

    # Droop output
    self.pd = Algeb(info='Pref plus under speed times gain',
                    unit='p.u.',
                    tex_name="P_d",
                    v_str='u * tm0',
                    e_str='u*(wd + pref + paux) * gain - pd')

    # Lag with anti-windup (replaces manual LG_x and LG_lim)
    self.LAG = LagAntiWindup(u=self.pd,
                             K=1,
                             T=self.T1,
                             lower=self.VMIN,
                             upper=self.VMAX)

    # Lead-lag block (replaces manual LL_x and LL_y)
    self.LL = LeadLag(u=self.LAG_y,
                      T1=self.T2,
                      T2=self.T3)

    # Output equation
    self.pout.e_str = '(LL_y + Dt * wd) - pout'
```

**Key observations:**
- `LagAntiWindup` encapsulates the lag transfer function with anti-windup
- `LeadLag` encapsulates the lead-lag transfer function
- Block outputs follow the pattern `BlockName_y`
- The block-based code is shorter and easier to validate against the diagram

## Comparison of Approaches

| Aspect | Equation-Based | Block-Based |
|--------|---------------|-------------|
| **Lines of code** | More | Fewer |
| **Readability** | Requires understanding equations | Matches block diagram |
| **Flexibility** | Full control | Limited to block capabilities |
| **Error risk** | Higher (manual equations) | Lower (validated blocks) |
| **Performance** | Same | Same |

## Guidelines

- **Use blocks** when the model follows a standard control block diagram
- **Use equations** when implementing non-standard transfer functions or when blocks don't exist
- **Combine both** approaches when needed - blocks can be mixed with custom equations

## Complete Source Code

The complete implementations can be found in:
- Equation-based: `andes/models/governor/tgov1.py` (class `TGOV1ModelAlt`)
- Block-based: `andes/models/governor/tgov1.py` (class `TGOV1Model`)

## See Also

- {doc}`../components/blocks` - Available transfer function blocks
- {doc}`../components/discrete` - Discrete components including AntiWindup
- {doc}`example-ieeest` - More complex example with input selection
