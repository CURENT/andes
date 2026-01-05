# DAE Formulation

ANDES models power systems as a set of Differential-Algebraic Equations (DAEs).

## Mathematical Form

The general DAE system is:

$$\mathbf{M} \dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{y})$$

$$\mathbf{0} = \mathbf{g}(\mathbf{x}, \mathbf{y})$$

Where:
- $\mathbf{x}$ are differential (state) variables
- $\mathbf{y}$ are algebraic variables
- $\mathbf{M}$ is the mass matrix (often identity)
- $\mathbf{f}$ are differential equations
- $\mathbf{g}$ are algebraic equations

## Variable Types

### State Variables ($\mathbf{x}$)

Describe dynamic behavior—change continuously over time.

Examples:
- Generator rotor angle ($\delta$)
- Generator speed deviation ($\omega - 1$)
- Exciter states
- Governor states

```python
# In model definition
self.omega = State(v_str='1.0',
                   e_str='(Tm - Te - D*(omega-1)) / M')
```

### Algebraic Variables ($\mathbf{y}$)

Satisfy instantaneous constraints—can change discontinuously.

Examples:
- Bus voltage magnitude ($V$)
- Bus voltage angle ($\theta$)
- Generator electrical power ($P_e$)
- Current injections

```python
# In model definition
self.v = Algeb(v_str='1.0',
               e_str='P - v * sum(Y * V * cos(theta))')
```

## Power Flow vs Dynamic

### Power Flow (Steady-State)

All derivatives are zero: $\dot{\mathbf{x}} = 0$

The system becomes purely algebraic:
$$\mathbf{0} = \mathbf{f}(\mathbf{x}, \mathbf{y})$$
$$\mathbf{0} = \mathbf{g}(\mathbf{x}, \mathbf{y})$$

Solved using Newton-Raphson iteration.

### Time-Domain Simulation

Differential equations are integrated over time using implicit methods (trapezoidal rule).

At each time step, discretized equations become:
$$\mathbf{x}_{n+1} - \mathbf{x}_n = \frac{h}{2}(\mathbf{f}_{n+1} + \mathbf{f}_n)$$

Combined with algebraic constraints, solved iteratively.

## Jacobian Matrices

For Newton-Raphson iteration, we need:

$$\mathbf{J} = \begin{bmatrix} \frac{\partial \mathbf{f}}{\partial \mathbf{x}} & \frac{\partial \mathbf{f}}{\partial \mathbf{y}} \\ \frac{\partial \mathbf{g}}{\partial \mathbf{x}} & \frac{\partial \mathbf{g}}{\partial \mathbf{y}} \end{bmatrix}$$

ANDES computes these automatically from symbolic equations:

```python
ss.dae.fx    # df/dx
ss.dae.fy    # df/dy
ss.dae.gx    # dg/dx
ss.dae.gy    # dg/dy
```

## Equation Conventions

### Differential Equations

Written in the form: $M \dot{x} = f(x, y)$

```python
# Swing equation: M * d(omega)/dt = Tm - Te - D*(omega-1)
self.omega = State(e_str='Tm - Te - D*(omega - 1)')
# M is specified separately
```

### Algebraic Equations

Written in the form: $0 = g(x, y)$

```python
# Power balance: 0 = P_gen - P_load - P_flow
self.P = Algeb(e_str='Pg - Pl - Pflow')
```

## Initialization

Before simulation, initial values must satisfy:

$$\mathbf{0} = \mathbf{f}(\mathbf{x}_0, \mathbf{y}_0)$$
$$\mathbf{0} = \mathbf{g}(\mathbf{x}_0, \mathbf{y}_0)$$

ANDES supports two initialization methods:

### Explicit (`v_str`)

Direct calculation of initial value:

```python
self.omega = State(v_str='1.0')  # omega starts at 1.0 pu
```

### Implicit (`v_iter`)

Solve iteratively using Newton-Krylov:

```python
self.Efd = Algeb(v_iter='Vf - Efd')  # Solve Efd such that Vf - Efd = 0
```

## Handling Discontinuities

Power systems have discontinuous behavior:
- Limits (saturation, rate limits)
- Switching (breaker operations)
- Dead bands

ANDES uses **discrete components** that export flags:

```python
self.lim = Limiter(u=self.x, lower=-1, upper=1)
# Exports: lim.zi (within), lim.zl (below), lim.zu (above)

# Use in equation
self.y = Algeb(e_str='x * lim_zi + upper * lim_zu + lower * lim_zl')
```

## Mass Matrix

For most models, $\mathbf{M} = \mathbf{I}$ (identity).

Some models use non-unity mass matrix entries:

```python
self.omega = State(e_str='(Tm - Te) / M')
# Equivalent to: M * d(omega)/dt = Tm - Te
```

## Eigenvalue Analysis

At equilibrium, linearize around $(\mathbf{x}_0, \mathbf{y}_0)$:

$$\Delta \dot{\mathbf{x}} = \mathbf{A}_s \Delta \mathbf{x}$$

Where the state matrix is:

$$\mathbf{A}_s = \mathbf{f}_x - \mathbf{f}_y \mathbf{g}_y^{-1} \mathbf{g}_x$$

Eigenvalues of $\mathbf{A}_s$ determine stability.

## See Also

- {doc}`atoms` - v-provider and e-provider concepts
- {doc}`framework-overview` - Symbolic framework overview
- {doc}`../components/variables` - Variable types in detail
- {doc}`../components/discrete` - Discrete components
