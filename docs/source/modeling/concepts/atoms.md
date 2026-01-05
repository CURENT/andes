# Atomic Types

ANDES contains three fundamental types of atom classes for building DAE models:
**parameters**, **variables**, and **services**. These atom classes share common
interfaces that enable interoperability and composability when defining model equations.

## Value Provider (v-provider)

A **value provider** class (or *v-provider* for short) is any class with a member
attribute named `v`, which should be a list or a 1-dimensional NumPy array of values.
The `v` attribute contains the numerical values that will be substituted into equations
during computation.

```{note}
All atom classes are v-providers. Every parameter, variable, and service instance
must contain values accessible through the `v` attribute.
```

### How v-providers Work

When you define parameters in a model:

```python
self.v0 = NumParam(default=1.0, info='Initial voltage')
self.b = NumParam(default=10.0, info='Susceptance')

# At runtime, these might contain:
# self.v0.v = np.array([1.0, 1.05, 1.1])
# self.b.v  = np.array([10., 10., 10.])
```

When these parameters are used in an equation string:

```python
self.v = ExtAlgeb(model='Bus', src='v',
                  indexer=self.bus,
                  e_str='v0 ** 2 * b')
```

During equation evaluation, `v0` and `b` are substituted with their respective
`v` arrays (`self.v0.v` and `self.b.v`). The symbolic expression `v0 ** 2 * b`
becomes a vectorized numerical computation.

### Interoperability Through the v Interface

The shared `v` interface enables seamless substitution between different atom types.
Consider this parameter definition:

```python
self.v0 = NumParam(default=1.0)
```

You could replace it with a service that computes the value:

```python
self.v0 = ConstService(v_str='1.0')
```

**Equations using `v0` continue to work without modification**, because both
`NumParam` and `ConstService` provide values through the same `v` attribute.
This design pattern allows:

- Parameters to be replaced by computed services
- Services to reference other services or parameters
- Variables to be used in place of parameters in equation strings

## Equation Provider (e-provider)

An **equation provider** class (or *e-provider*) is any class with a member
attribute named `e`, which should be a 1-dimensional array. The `e` array stores
equation evaluation results that are accumulated into the numerical DAE system
at addresses specified by the `a` attribute.

```{note}
Currently, only variables (differential and algebraic) are e-providers. Parameters
and services provide values but do not contribute equations to the DAE.
```

### How e-providers Work

When you define an external variable that links to bus voltage:

```python
self.v = ExtAlgeb(model='Bus', src='v',
                  indexer=self.bus,
                  e_str='v0 ** 2 * b')
```

Three things happen:

1. **Address retrieval**: The addresses of the corresponding voltage variables
   are retrieved into `self.v.a`
2. **Equation evaluation**: The expression `v0 ** 2 * b` is evaluated and stored
   in `self.v.e`
3. **DAE assembly**: Values in `self.v.e` are summed into the system DAE array
   at the addresses in `self.v.a`

### The v-e-a Triad for Variables

Variables have three key attributes that work together:

| Attribute | Purpose | Description |
|-----------|---------|-------------|
| `v` | Values | Current values of the variable (solution) |
| `e` | Equations | Equation contributions (residuals) |
| `a` | Addresses | Indices into the system DAE arrays |

This triad enables the numerical integration scheme:
- **`v`** holds the current state/algebraic values being solved
- **`e`** accumulates equation right-hand-sides from this model
- **`a`** specifies where in the global DAE this variable lives

## Summary

| Atom Type | v-provider | e-provider | Purpose |
|-----------|:----------:|:----------:|---------|
| Parameter | Yes | No | Input data for models |
| Service | Yes | No | Intermediate calculations |
| Variable | Yes | Yes | DAE unknowns and equations |

The v-provider interface creates a uniform way to access values, enabling:
- Symbolic equation processing that works with any value source
- Easy substitution between parameters, services, and variables
- Clean separation between data definition and equation evaluation

## See Also

- {doc}`../components/parameters` - Parameter types and usage
- {doc}`../components/variables` - Variable types and the v-e-a attributes
- {doc}`../components/services` - Service types for intermediate calculations
