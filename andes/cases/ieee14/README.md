# IEEE 14-Bus Systems

The folder contains many variants of the IEEE 14-bus system that are created
during model development. One cawn find cases containing the needed model and modify
from them.
## Power flow data

Power flow data is created by J. Conto obtained from the [Google drive
link](https://drive.google.com/drive/folders/0B7uS9L2Woq_7YzYzcGhXT2VQYXc).

The base power flow file is ``ieee14.raw``. Almost all other cases in this
folder are created by adding models to it.

## Dynamic data

The base dynamic data is created by H. Cui for ANDES, given in ``ieee14.dyr``.

## Special cases

### Contingency cases

- `ieee14_gentrip.xlsx` contains data for a generator trip using `Toggler`

- `ieee14_linetrip.xlsx` contains data for a line trip

- `ieee14_fault.xlsx` contains data for a three-phase-to-ground fault using `Fault`

### Renewable energy models

- `ieee14_wt3.xlsx` contains one generic Type-3 wind turbine

- `ieee14_wt3n.xlsx` has (almost) all generators represented by Type-3 wind
  turbines

- `ieee14_solar.xlsx` contains a generic solar PV device, which can also be used
  to model a Type-4 wind turbine

- `ieee14_pvd1.xlsx` contains WECC distributed PV

- `ieee14_dgprct1.xlsx` contains the WECC distributed PV model with IEEE
  1547.2018-based voltage and frequency protection

### Playback and Timeseries
- `ieee14_plbvfu1.xlsx` contains data for setting up a playback V-f generator

- `ieee14_timeseries.xlsx` contains data for using time series from `pqts.xlsx`
  for a PQ load. Note that one needs to convert PQ to constant power load for
  simulation.

### Other
- `ieee14_jumper.xlsx` contains data for a jumper device that connects two buses
  without impedance.
