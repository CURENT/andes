# Gallery

Application examples showcasing ANDES for power system research.
Each example is a self-contained notebook that you can download and extend for your own studies.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Critical Clearing Time Mapping
:link: critical-clearing-time
:link-type: doc
:img-top: images/critical-clearing-time.png

Sweep fault durations across all buses in the Kundur two-area system
and map the transient stability boundary as a heatmap.

+++
Kundur 2-area | PFlow, TDS, reinit
:::

:::{grid-item-card} Low-Inertia Frequency Stability
:link: low-inertia-freq
:link-type: doc
:img-top: images/low-inertia-freq.png

Scale generator inertia to emulate rising renewable penetration on
the IEEE 14-bus system. Quantify how RoCoF and frequency nadir degrade.

+++
IEEE 14-bus | PFlow, TDS
:::

:::{grid-item-card} Forced Oscillation Source Localization
:link: forced-oscillation
:link-type: doc
:img-top: images/forced-oscillation.png

Inject a periodic governor disturbance on the WECC 179-bus system
and locate the source using energy flow and phase analysis from PMU data.

+++
WECC 179-bus | PFlow, TDS, FFT
:::

:::{grid-item-card} Physics-Informed Surrogate (PINN)
:link: pinn-surrogate
:link-type: doc
:img-top: images/pinn-surrogate.png

Extract symbolic DAE equations from ANDES, eliminate algebraic variables,
and train a parameterized PINN surrogate. No hand-coded physics in PyTorch.

+++
SMIB | SymPy, PyTorch, PINN
:::

:::{grid-item-card} RL Oscillation Damping on SMIB
:link: smib-oscillation
:link-type: doc
:img-top: images/smib-oscillation.png

Train a PPO agent to damp post-fault rotor oscillations using the
Gymnasium-compatible AndesEnv. Zero domain knowledge, pure simulation experience.

+++
SMIB | PFlow, TDS, AndesEnv, PPO
:::

::::

## Running the Examples

All examples use built-in test cases and can be run directly after installing ANDES:

```bash
pip install andes[dev]    # includes matplotlib, pandas
pip install scikit-learn  # optional, for ML extensions
```

Each notebook is designed as a **template**: after following the walkthrough, look for the
"Extend This Example" section at the end for ideas on adapting it to your own research.

```{toctree}
:maxdepth: 1
:hidden:

critical-clearing-time
low-inertia-freq
forced-oscillation
pinn-surrogate
smib-oscillation
```
