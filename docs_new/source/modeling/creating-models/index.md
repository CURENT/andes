# Creating Models

Step-by-step guide to implementing new device models in ANDES.

````{only} html
## Examples Overview

The examples progress from simple to advanced:

| Example | Model | Complexity | Key Concepts |
|---------|-------|------------|--------------|
| {doc}`example-static` | Shunt | Introductory | Static model, constant impedance behavior |
| {doc}`example-dynamic` | BusFreq | Introductory | Dynamic model, transfer function blocks, signal processing |
| {doc}`example-tgov1` | TGOV1 | Intermediate | Two implementation approaches (equation vs block), limiters |
| {doc}`example-ieeest` | IEEEST | Advanced | External services, switchers, signal processing chain |

## Workflow

1. **Define ModelData class** - Declare parameters
2. **Define Model class** - Declare variables, services, equations
3. **Register with System** - Add to model registry
4. **Generate code** - Run `andes prepare`
5. **Test** - Validate against reference implementations
````

```{toctree}
:maxdepth: 1

model-structure
example-static
example-dynamic
example-tgov1
example-ieeest
testing-models
```
