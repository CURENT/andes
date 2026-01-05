# Modeling Guide

This section provides an in-depth explanation of how the ANDES framework is set up for symbolic modeling and numerical simulation. It aims to help developers understand the internal architecture and implement customized DAE models.

```{toctree}
:maxdepth: 2

concepts/index
components/index
creating-models/index
```

## Overview

### Concepts
The hybrid symbolic-numeric framework, DAE formulation, and system architecture that underpin ANDES. Start here to understand how ANDES processes symbolic equations into executable numerical code.

### Components
Building blocks for models: parameters, variables, services, discrete components, and transfer function blocks. Each component type is documented with its API reference and usage patterns.

### Creating Models
Step-by-step guide to implementing new power system device models, from simple static models to complex dynamic controllers.

## Who This Is For

This section is for developers who want to:
- Understand how ANDES models work internally
- Implement new power system device models
- Extend ANDES capabilities with custom components

For users who want to:
- Run simulations: see the {doc}`../tutorials/index` section
- Inspect model equations: see the {doc}`../tutorials/inspecting-models` tutorial
