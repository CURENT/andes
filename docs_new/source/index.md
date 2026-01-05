# ANDES Documentation

**Python Software for Symbolic Power System Modeling and Numerical Analysis**

```{image} images/sponsors/CURENT_Logo_NameOnTrans.png
:alt: CURENT Logo
:width: 300px
```

ANDES is an open-source Python library for power system modeling, computation, analysis, and control. It supports:

- **Power flow** calculation
- **Time-domain simulation** (transient stability)
- **Eigenvalue analysis** (small-signal stability)
- Symbolic-numeric framework for rapid model prototyping
- Full second-generation renewable energy models

---

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Complete learning path from installation through advanced analysis: power flow, time-domain simulation, eigenvalue analysis, parameter sweeps, and contingency studies.
:::

:::{grid-item-card} Modeling Guide
:link: modeling/index
:link-type: doc

Framework internals: inspect model equations, understand the symbolic-numeric framework, and create new device models.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

CLI commands, configuration options, model reference (auto-generated), and API documentation.
:::

::::

---

## Quick Install

::::{tab-set}

:::{tab-item} conda
```bash
conda install -c conda-forge andes
```
:::

:::{tab-item} pip
```bash
pip install andes
```
:::

:::{tab-item} uv
```bash
uv pip install andes
```
:::

:::{tab-item} development
```bash
git clone https://github.com/CURENT/andes
cd andes
pip install -e .[dev]
```
:::

::::

## Quick Example

```python
import andes

# Load a test case and run power flow
ss = andes.load(andes.get_case('ieee14/ieee14.json'))
ss.PFlow.run()

# Run time-domain simulation
ss.TDS.run()

# Plot generator speeds
ss.TDS.plt.plot(ss.GENROU.omega)
```

---

## Learning Paths

::::{grid} 1 3 3 3
:gutter: 2

:::{grid-item-card} New User
1. {doc}`tutorials/01-installation`
2. {doc}`tutorials/02-first-simulation`
3. {doc}`tutorials/03-power-flow`
:::

:::{grid-item-card} Power System Analyst
1. Complete New User path
2. {doc}`tutorials/05-data-and-formats`
3. {doc}`tutorials/07-eigenvalue-analysis`
4. {doc}`tutorials/08-parameter-sweeps`
:::

:::{grid-item-card} Model Developer
1. Complete Analyst path
2. {doc}`tutorials/inspecting-models`
3. {doc}`modeling/concepts/framework-overview`
4. {doc}`modeling/creating-models/index`
:::

::::

---

## Citation

If you use ANDES in your research, please cite:

> H. Cui, F. Li and K. Tomsovic, "Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis," *IEEE Transactions on Power Systems*, vol. 36, no. 2, pp. 1373-1384, March 2021. [DOI: 10.1109/TPWRS.2020.3017019](https://doi.org/10.1109/TPWRS.2020.3017019)

---

```{toctree}
:maxdepth: 2
:hidden:

tutorials/index
modeling/index
reference/index
verification/index
```
