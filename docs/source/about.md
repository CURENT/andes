# About ANDES

ANDES is an open-source Python library for power system modeling, computation, analysis, and control. It serves as the dynamic simulation engine for the CURENT Large Scale Testbed (LTB), supporting power flow calculation, transient stability simulation, and small-signal stability analysis for transmission systems.

## Symbolic-Numeric Framework

At its core, ANDES implements a hybrid symbolic-numeric framework that separates model description from numerical implementation. Device models are written as symbolic equations in Python, which ANDES processes using SymPy to generate optimized numerical code automatically. This allows researchers to prototype new models without manually deriving Jacobians or writing simulation code. The generated code is cached and reused, so the symbolic overhead is incurred only once.

## What's Included

- **Comprehensive model library** — synchronous generators, exciters, turbine governors, PSS, and full second-generation renewable models (solar PV, Type 3 and Type 4 wind) following WECC specifications
- **Verified results** — models produce identical time-domain simulation results against commercial software for IEEE 14-bus and NPCC systems with GENROU and multiple controllers
- **Industry file formats** — PSS/E raw and dyr parsing for direct use of standard case files
- **Production performance** — 20-second transient simulation of a 2000-bus system completes in seconds on a typical desktop, achieved through vectorized NumPy operations and sparse matrix handling

## Citation

If ANDES is used in your research, please cite:

> H. Cui, F. Li and K. Tomsovic, "Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis," *IEEE Transactions on Power Systems*, vol. 36, no. 2, pp. 1373-1384, March 2021. [DOI: 10.1109/TPWRS.2020.3017019](https://doi.org/10.1109/TPWRS.2020.3017019)

## Acknowledgments

This work was supported in part by the Engineering Research Center Program of the National Science Foundation and the Department of Energy under NSF Award Number EEC-1041877 and the CURENT Industry Partnership Program.

ANDES is developed and maintained by [Hantao Cui](https://cui.eecps.com). See the [GitHub repository](https://github.com/CURENT/andes) for a full list of contributors.

## License

ANDES is released under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.html).
