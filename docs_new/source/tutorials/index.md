# Tutorials

Step-by-step guides to learn ANDES, from installation to advanced analysis workflows.

These tutorials are designed to be followed in sequence for new users. Each tutorial builds on concepts from previous ones, gradually introducing more sophisticated analysis techniques. Experienced users can jump directly to topics of interest.

```{toctree}
:maxdepth: 1

01-installation
02-first-simulation
03-power-flow
04-time-domain
05-data-and-formats
06-plotting-results
07-eigenvalue-analysis
08-parameter-sweeps
09-contingency-analysis
10-dynamic-control
11-frequency-response
inspecting-models
```

## Tutorial Overview

| # | Tutorial | Description |
|---|----------|-------------|
| 01 | {doc}`01-installation` | Install ANDES and verify the installation |
| 02 | {doc}`02-first-simulation` | Load a case, run power flow and TDS, plot results |
| 03 | {doc}`03-power-flow` | Newton-Raphson power flow analysis |
| 04 | {doc}`04-time-domain` | Time-domain simulation with disturbances |
| 05 | {doc}`05-data-and-formats` | Load PSS/E, MATPOWER, Excel files; modify parameters |
| 06 | {doc}`06-plotting-results` | Visualize and export simulation results |
| 07 | {doc}`07-eigenvalue-analysis` | Small-signal stability and root locus analysis |
| 08 | {doc}`08-parameter-sweeps` | Batch processing and parallel execution |
| 09 | {doc}`09-contingency-analysis` | N-1 contingency screening and CCT calculation |
| 10 | {doc}`10-dynamic-control` | Multi-stage simulation with setpoint changes |
| 11 | {doc}`11-frequency-response` | Generator trips and load shedding |
| -- | {doc}`inspecting-models` | Examine model equations, variables, and services |

## Learning Paths

### New User
Start with tutorials 01-06 to learn the basics of ANDES:
- Installation and verification
- Running simulations
- Working with data files
- Visualizing results

### Power System Analyst
After completing the basics, continue with tutorials 07-11 for advanced analysis:
- Small-signal stability assessment
- Parameter sensitivity studies
- Contingency analysis
- Dynamic control studies

### Model Developer
For creating custom device models, see the {doc}`../modeling/index` section after completing these tutorials.
