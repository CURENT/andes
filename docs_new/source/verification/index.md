# Verification

This section presents the verification of the models and algorithms implemented in ANDES by comparing time-domain simulation results with commercial tools.

## Overview

ANDES produces identical results for the IEEE 14-bus and the NPCC systems with several models. For the CURENT WECC system, ANDES, TSAT, and PSS/E produce slightly different results. In fact, results from different tools can hardly match for large systems with a variety of dynamic models.

ANDES models are verified against:
- PSS/E simulation results
- TSAT simulation results
- Published benchmark data

## Test Cases

### IEEE 14-Bus System
- Compact test system with synchronous generators and exciters
- Verified against PSS/E and TSAT
- Results show excellent agreement

### NPCC System
- Northeast Power Coordinating Council test system
- Multi-area system with various generator models
- Verified against commercial software

### WECC System
- Western Electricity Coordinating Council test system
- Large-scale system with renewable generation
- Results show minor differences due to model implementation variations

## Verification Approach

1. **Model Parameters**: Use identical parameters across all tools
2. **Disturbance**: Apply the same fault or disturbance
3. **Time Step**: Use comparable integration time steps
4. **Comparison Variables**: Compare rotor angles, speeds, and bus voltages

## Detailed Results

The following notebooks contain detailed verification results with comparison plots:

```{toctree}
:maxdepth: 1

andes-ieee14-verification
andes-npcc-verification
andes-wecc-verification
```
